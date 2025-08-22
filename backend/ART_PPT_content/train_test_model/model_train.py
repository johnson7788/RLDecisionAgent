"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import argparse
import asyncio
import fnmatch
import json
import os
import wandb
import weave
from dotenv import load_dotenv

import art
from art.rewards import ruler_score_group
from art.utils import iterate_dataset

from .benchmarks.generate_benchmarks import calculate_beat_comp, generate_val_groups
from .rollout import McpScenario, rollout
from .experiments_config import models
load_dotenv()

if os.getenv("WANDB_API_KEY"):
    print("Initializing Weave 和 Wandb")
    wandb.init(
        project="mcp_alphavantage",
        entity="johnson-"
    )
    # weave.init("mcp_alphavantage")


async def train_mcp_agent(model: art.TrainableModel, use_skypilot: bool = False):
    """Example training function that creates AlphaMcpServer and passes it in scenarios."""
    print(f"[INFO] 开始进行 train_mcp_agent 的准备工作")
    load_dotenv()
    print(f"[INFO] 环境变量加载完成")

    gpt_4o = art.Model(
        name="gpt-4o",
        project=model.project,
        inference_model_name="gpt-4o-mini",
        inference_api_key=os.getenv("OEPNAI_API_KEY"),
        inference_base_url="http://localhost:6688",
    )
    print(f"[INFO] GPT-4o 模型实例创建完成: {gpt_4o}")

    # Get configuration from model config or use defaults
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model config is required")
    print(f"[INFO] 模型配置加载成功")

    max_turns = config.max_turns
    trajectories_per_group = config.trajectories_per_group
    groups_per_step = config.groups_per_step
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    eval_steps = config.eval_steps
    ruler_judge_model = config.ruler_judge_model
    training_dataset_size = config.training_dataset_size
    mcp_server_name = config.mcp_server_name
    print(f"[INFO] 配置参数: max_turns={max_turns}, trajectories_per_group={trajectories_per_group}, "
          f"groups_per_step={groups_per_step}, num_epochs={num_epochs}, learning_rate={learning_rate}, "
          f"eval_steps={eval_steps}, training_dataset_size={training_dataset_size}, mcp_server_name={mcp_server_name}")

    # Load server params dynamically based on config
    try:
        print(f"[INFO] 尝试导入 MCP Server 配置: {mcp_server_name}")
        server_params_module = __import__(
            f"{mcp_server_name}.server_params",
            fromlist=["server_params"],
        )
        server_params = server_params_module.server_params
        print(f"[INFO] MCP Server 配置导入成功")
    except ImportError:
        raise ValueError(
            f"Could not import server_params for MCP server: {mcp_server_name}"
        )

    # Load pre-split scenarios from scenarios directory
    scenarios_dir = f"{mcp_server_name}/scenarios"
    print(f"[INFO] 加载训练和验证场景文件 from {scenarios_dir}")

    with open(f"{scenarios_dir}/train.jsonl") as f:
        raw_train_scenarios = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"[INFO] 已加载 {len(raw_train_scenarios)} 条训练场景")

    with open(f"{scenarios_dir}/val.jsonl") as f:
        raw_val_scenarios = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"[INFO] 已加载 {len(raw_val_scenarios)} 条验证场景")

    # Limit training dataset size if specified in config
    if training_dataset_size and training_dataset_size < len(raw_train_scenarios):
        raw_train_scenarios = raw_train_scenarios[:training_dataset_size]
        print(f"[INFO] 限制训练数据集大小为 {training_dataset_size}")

    # Backend initialization
    if use_skypilot:
        print(f"[INFO] 使用 Skypilot 远端服务进行训练")
        from art.skypilot.backend import SkyPilotBackend

        backend = await SkyPilotBackend().initialize_cluster(
            cluster_name="mcp-agent-training", gpu="H100-SXM"
        )
        print(f"[INFO] Skypilot 集群初始化完成")
    else:
        from art.local.backend import LocalBackend
        print(f"[INFO] 使用本地 Backend 进行训练")
        backend = LocalBackend()

    print(f"[INFO] 开始注册模型到 Backend")
    await model.register(backend)
    print(f"[INFO] 模型注册成功")

    # Create McpScenario objects for training and validation
    print(f"[INFO] 开始生成训练场景对象")
    train_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=max_turns,
        )
        for scenario in raw_train_scenarios
    ]
    print(f"[INFO] 训练场景对象生成完成，共 {len(train_scenarios)} 个")

    print(f"[INFO] 开始生成验证场景对象")
    val_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=max_turns,
        )
        for scenario in raw_val_scenarios
    ]
    print(f"[INFO] 验证场景对象生成完成，共 {len(val_scenarios)} 个")

    print(f"[INFO] 创建训练数据迭代器 train_iterator")
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        initial_step=await model.get_step(),  # Resume from checkpoint
    )
    print(f"[INFO] 训练数据迭代器创建完成")

    print(f"[INFO] 生成控制组 validation groups")
    control_groups = await generate_val_groups(gpt_4o, val_scenarios)
    print(f"[INFO] 控制组生成完成")

    # Main training loop
    for batch in train_iterator:
        print(f"[INFO] 开始处理训练步 {batch.step}, 收集 trajectory groups")
        extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ["OPENAI_API_KEY"]}
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, False)
                    for _ in range(trajectories_per_group)
                )
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
            after_each=lambda group: ruler_score_group(
                group,
                judge_model=ruler_judge_model,
                extra_litellm_params=extra_litellm_params,
                debug=True,
                swallow_exceptions=True,
            ),
        )
        print(f"[INFO] 收集训练轨迹完成，开始训练模型")

        if batch.step % eval_steps == 0:
            print(f"[INFO] 步 {batch.step} 达到验证间隔，开始验证")
            val_groups = await generate_val_groups(model, val_scenarios)
            await calculate_beat_comp(val_groups, control_groups, control_first=True)
            await calculate_beat_comp(val_groups, control_groups, control_first=False)
            await model.log(val_groups, split="val")
            print(f"[INFO] 验证完成并已记录日志")

        await model.train(groups, config=art.TrainConfig(learning_rate=learning_rate))
        print(f"[INFO] 步 {batch.step} 模型训练完成")


def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(description="Train MCP agent models.")
    parser.add_argument(
        "--models",
        type=str,
        default="ppt_agent_01",
        help="使用的训练模型的配置信息，从experiments_config中读取",
    )
    parser.add_argument(
        "--use-skypilot",
        action="store_true",
        help="Whether to use SkyPilot backend instead of local backend.",
    )

    args = parser.parse_args()

    if args.models not in models:
        # Check for wildcard patterns
        matching_keys = [
            key
            for key in models.keys()
            if fnmatch.fnmatch(key, args.models.replace("%", "*"))
        ]
        if matching_keys:
            if len(matching_keys) > 1:
                print(
                    f"Multiple models matched pattern '{args.models}': {', '.join(sorted(matching_keys))}"
                )
                print("Please specify a single model key.")
                return
            model_key = matching_keys[0]
        else:
            print(
                f"Unknown model key: {args.models}. Valid keys: {', '.join(sorted(models.keys()))}"
            )
            return
    else:
        model_key = args.models

    model = models[model_key].model_copy(deep=True)
    print(f"Using model configuration: {model_key} ({model.name})")

    print(f"Starting MCP agent training..., use_skypilot: {args.use_skypilot}")
    try:
        asyncio.run(train_mcp_agent(model, use_skypilot=args.use_skypilot))
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise  # Re-raise the exception to ensure it reaches the user


if __name__ == "__main__":
    main()
