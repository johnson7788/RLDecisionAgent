from pydantic import BaseModel
import art
from mcp_search.server_params import server_params as search_server_params

# 简单配置类
class McpPolicyConfig(BaseModel):
    max_turns: int = 5
    max_tokens: int = 2048
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    mcp_server_name: str = "mcp_search"
    # 训练相关配置
    trajectories_per_group: int = 7           # 每个训练组的轨迹数量
    groups_per_step: int = 4                  # 每步训练的组数
    learning_rate: float = 1e-6               # 学习率
    eval_steps: int = 1                        # 每多少步进行一次评估
    val_set_size: int = 8                       # 验证集大小
    training_dataset_size: int = 4            # 训练数据集大小
    num_epochs: int = 4                        # 训练轮数
    ruler_judge_model: str = "openai/o3"  # 用于 RULER 重评分的模型
    minimum_reward_std_dev: float = 0.0        # 奖励的最小标准差
    training_dataset_seed: int | None = None   # 随机种子，用于训练数据采样

    # 模型分支配置，用于从已有模型 fork
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # 是否对奖励进行缩放
    scale_rewards: bool = True

MCP_SERVERS = {
    "mcp_search": {
        "server_params": search_server_params,
        "val_data": "mcp_search/scenarios/val.jsonl"
    }
}

# 定义模型集合,重置一些默认的配置
models: dict = {
    "ppt_agent_01": art.TrainableModel(
        name="ppt_agent_01",
        project="ppt_project_01",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        config=McpPolicyConfig(num_epochs=4),
    ),

    "mcp-14b-001": art.TrainableModel(
        name="mcp-14b-001",
        project="mcp-agent-training",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(num_epochs=4),
    ),

    "mcp-14b-alpha-001": art.TrainableModel(
        name="mcp-14b-alpha-001",
        project="mcp_alphavantage",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(num_epochs=4, mcp_server_name="mcp_search"),
    ),

    "mcp-14b-alpha-002": art.TrainableModel(
        name="mcp-14b-alpha-002",
        project="mcp_alphavantage",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(num_epochs=8, mcp_server_name="mcp_search"),
    ),

    "mcp-14b-alpha-003": art.TrainableModel(
        name="mcp-14b-alpha-003",
        project="mcp_alphavantage",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(num_epochs=8, mcp_server_name="mcp_search"),
    ),

    "mcp-14b-alpha-004": art.TrainableModel(
        name="mcp-14b-alpha-004",
        project="mcp_alphavantage",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(num_epochs=8, mcp_server_name="mcp_search", learning_rate=1e-6),
    ),

    "mcp-14b-ball-001": art.TrainableModel(
        name="mcp-14b-ball-001",
        project="mcp_balldontlie",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(num_epochs=4, mcp_server_name="mcp_search"),
    ),
}
