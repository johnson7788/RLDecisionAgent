import argparse
import asyncio
import json
import os
import random
from typing import List

# import weave
from dotenv import load_dotenv
import art
from art.local import LocalBackend
from art.rewards.ruler import ruler_score_group
from rollout import McpScenario, rollout
from experiments_config import MCP_SERVERS

load_dotenv()

random.seed(42)

# Initialize the server
backend = LocalBackend()


async def generate_val_groups(
    model: art.Model, val_scenarios: List[McpScenario]
) -> list[art.TrajectoryGroup]:
    print(f"正在为模型生成验证组: {model.name}")
    groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(model, val_scenarios[i]) for _ in range(4))
            for i in range(len(val_scenarios))
        ),
        pbar_desc=f"gather {model.name}",
        max_exceptions=1,
    )
    print(f"为模型生成了 {len(groups)} 个验证组: {model.name}")
    return groups


async def calculate_beat_comp(
    groups: list[art.TrajectoryGroup],
    control_groups: list[art.TrajectoryGroup],
    control_first: bool = True,
):
    print(f"正在计算击败对比, control_first: {control_first}")
    promises = []

    if control_groups is not None:
        for i in range(len(groups)):
            for j in range(len(groups[i].trajectories)):
                trajectories = [
                    control_groups[i].trajectories[j],
                    groups[i].trajectories[j],
                ]
                group = art.TrajectoryGroup(
                    trajectories if control_first else reversed(trajectories)
                )

                async def score_group(group_idx: int, trajectory_idx: int):
                    print(f"正在评分组 {group_idx}, 轨迹 {trajectory_idx}")
                    judge_model = "openai/o3-mini"
                    extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ["OPENAI_API_KEY"]}
                    print(f"使用的评判模型为 {judge_model}")
                    scored_group = await ruler_score_group(
                        group,
                        judge_model= judge_model,
                        extra_litellm_params=extra_litellm_params,
                        debug=True,
                    )

                    if control_first:
                        control_score = scored_group.trajectories[0].reward
                        benchmark_score = scored_group.trajectories[1].reward
                    else:
                        benchmark_score = scored_group.trajectories[0].reward
                        control_score = scored_group.trajectories[1].reward

                    reward_diff = benchmark_score - control_score
                    print(f"组 {group_idx}, 轨迹 {trajectory_idx} 的奖励差异: {reward_diff}")

                    metric_name = (
                        "beat_comp" if control_first else "beat_comp_control_last"
                    )

                    if reward_diff > 0.1:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 1
                    elif reward_diff < -0.1:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 0
                    else:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 0.5

                promises.append(score_group(i, j))

    await asyncio.gather(*promises)
    print(f"完成计算击败对比, control_first: {control_first}")


async def log_comparison_model(
    comparison_model: art.Model,
    val_scenarios: List[McpScenario],
    control_groups: list[art.TrajectoryGroup] | None = None,
) -> list[art.TrajectoryGroup]:
    print(f"正在记录对比模型: {comparison_model.name}")
    groups = await generate_val_groups(comparison_model, val_scenarios)

    if control_groups is not None:
        print(f"正在为 {comparison_model.name} 计算击败对比")
        await calculate_beat_comp(groups, control_groups, control_first=True)
        await calculate_beat_comp(groups, control_groups, control_first=False)

    await comparison_model.log(
        groups,
        split="val",
    )
    # print(f"正在将 {comparison_model.name} 推送到S3")
    # await backend._experimental_push_to_s3(
    #     comparison_model,
    # )
    print(f"完成记录对比模型: {comparison_model.name}")

    return groups


async def run_benchmarks(server: str = "mcp_alphavantage"):
    print(f"正在为服务器运行基准测试: {server}")
    mcp_configs = MCP_SERVERS[server]
    scenarios_path = mcp_configs["val_data"]
    server_params = mcp_configs["server_params"]

    # weave.init(server)
    # print(f"使用项目初始化Weave: {server}")

    # comparison models
    gpt_4o_mini = art.Model(
        name="gpt-4o-mini",
        project=server,
        inference_model_name="gpt-4o-mini",
        inference_base_url="http://localhost:6688",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    gpt_4o = art.Model(
        name="gpt-4o",
        project=server,
        inference_model_name="gpt-4o",
        inference_base_url="http://localhost:6688",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    gpt_41 = art.Model(
        name="gpt-4.1",
        project=server,
        inference_model_name="openai/gpt-4.1",
        inference_base_url="http://localhost:6688",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    o3 = art.Model(
        name="o3",
        project=server,
        inference_model_name="o3",
        inference_base_url="https://api.openai.com/v1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    o4_mini = art.Model(
        name="o4-mini",
        project=server,
        inference_model_name="o4-mini",
        inference_base_url="https://api.openai.com/v1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    deepseek_chat = art.Model(
        name="deepseek",
        project=server,
        inference_model_name="gpt-4o-mini",
        inference_base_url="http://localhost:6688",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    doubao_chat = art.Model(
        name="deepseek",
        project=server,
        inference_model_name="deepseek-r1-250528",
        inference_base_url="http://localhost:6688",
        inference_api_key=os.getenv("DOUBAO_API_KEY"),
    )
    sonnet_4 = art.Model(
        name="sonnet-4",
        project=server,
        inference_model_name="claude-sonnet-4",
        inference_base_url="http://localhost:6688",
        inference_api_key=os.getenv("CLAUDE_API_KEY"),
    )

    print(f"正在从以下位置加载场景: {scenarios_path}")
    with open(scenarios_path) as f:
        raw_val_scenarios = [json.loads(line.strip()) for line in f if line.strip()]
    val_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=2,
        )
        for scenario in raw_val_scenarios
    ]
    print(f"已加载 {len(val_scenarios)} 个验证场景")
    
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_41.register(backend)
    await o3.register(backend)
    await o4_mini.register(backend)
    await sonnet_4.register(backend)
    await deepseek_chat.register(backend)
    await doubao_chat.register(backend)
    print("已注册所有模型")

    print("正在使用gpt-4.1生成控制组")
    control_groups = await generate_val_groups(doubao_chat, val_scenarios)

    models = [gpt_4o_mini]
    for i, comparison_model in enumerate(models):
        print(f"正在处理模型 {i+1}/{len(models)}: {comparison_model.name}")
        await log_comparison_model(comparison_model, val_scenarios, control_groups)
    
    print("完成运行基准测试")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为MCP服务器生成基准测试")
    parser.add_argument(
        "--server",
        choices=["mcp_search", "mcp_balldontlie","mcp_caculator"],
        default="mcp_search",
        help="要进行基准测试的MCP服务器 (默认: mcp_search)",
    )
    args = parser.parse_args()

    print(f"开始使用服务器生成基准测试: {args.server}")
    asyncio.run(run_benchmarks(args.server))
    print("基准测试生成完成")