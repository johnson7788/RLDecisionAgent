# 导入 pydantic 的 BaseModel 用于定义配置数据类
from pydantic import BaseModel

# 导入 ART 库，用于训练和管理可训练模型
import art


# 定义 MCP 策略模型的配置类
class McpPolicyConfig(BaseModel):
    # 最大对话轮数
    max_turns: int = 5
    # 每轮对话允许的最大 token 数
    max_tokens: int = 2048

    # 基础模型名称，用于加载预训练模型
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # MCP 服务器配置
    mcp_server_name: str = "mcp_caculator"  # 默认使用 alphavantage 服务

    # 训练相关配置
    trajectories_per_group: int = 7           # 每个训练组的轨迹数量
    groups_per_step: int = 4                  # 每步训练的组数
    learning_rate: float = 1e-6               # 学习率
    eval_steps: int = 1                        # 每多少步进行一次评估
    val_set_size: int = 8                       # 验证集大小
    training_dataset_size: int = 16            # 训练数据集大小
    num_epochs: int = 80                        # 训练轮数
    ruler_judge_model: str = "deepseek/deepseek-chat"  # 用于 RULER 重评分的模型
    minimum_reward_std_dev: float = 0.0        # 奖励的最小标准差
    training_dataset_seed: int | None = None   # 随机种子，用于训练数据采样

    # 模型分支配置，用于从已有模型 fork
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # 是否对奖励进行缩放
    scale_rewards: bool = True


# 创建一个字典来管理不同的可训练模型
models: dict = {
    "mcp-7b-001": art.TrainableModel(
        name="mcp-7b-001",                       # 模型名称
        project="mcp-agent-training",            # 项目名
        base_model="Qwen/Qwen2.5-0.5B-Instruct", # 基础模型
        config=McpPolicyConfig(
            num_epochs=20,                       # 特殊指定该模型训练轮数为 20
        ),
    )
}

# 创建 14B 模型，通过复制 7B 模型的配置
models["mcp-14b-001"] = models["mcp-7b-001"].model_copy(deep=True)
models["mcp-14b-001"].name = "mcp-14b-001"
models["mcp-14b-001"].base_model = "Qwen/Qwen2.5-14B-Instruct"
models["mcp-14b-001"].config.num_epochs = 6  # 训练轮数设为 160

# 使用 alphavantage server 的 14B 模型
models["mcp-14b-alpha-001"] = models["mcp-7b-001"].model_copy(deep=True)
models["mcp-14b-alpha-001"].project = "mcp_alphavantage"  # 项目名
models["mcp-14b-alpha-001"].name = "mcp-14b-alpha-001"
models["mcp-14b-alpha-001"].config.mcp_server_name = "mcp_caculator"  # 指定服务器
models["mcp-14b-alpha-001"].config.num_epochs = 8               # 训练轮数设为 300

# 基于 alpha-001 生成多个副本模型，只修改名称
models["mcp-14b-alpha-002"] = models["mcp-14b-alpha-001"].model_copy(deep=True)
models["mcp-14b-alpha-002"].name = "mcp-14b-alpha-002"

models["mcp-14b-alpha-003"] = models["mcp-14b-alpha-001"].model_copy(deep=True)
models["mcp-14b-alpha-003"].name = "mcp-14b-alpha-003"

models["mcp-14b-alpha-004"] = models["mcp-14b-alpha-001"].model_copy(deep=True)
models["mcp-14b-alpha-004"].name = "mcp-14b-alpha-004"
models["mcp-14b-alpha-004"].config.learning_rate = 1e-6  # 特殊指定学习率

# 使用 balldontlie server 的 14B 模型
models["mcp-14b-ball-001"] = models["mcp-7b-001"].model_copy(deep=True)
models["mcp-14b-ball-001"].project = "mcp_balldontlie"
models["mcp-14b-ball-001"].name = "mcp-14b-ball-001"
models["mcp-14b-ball-001"].config.mcp_server_name = "mcp_balldontlie"
models["mcp-14b-ball-001"].config.num_epochs = 4
