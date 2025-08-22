from pydantic import BaseModel
import art
from mcp_search.server_params import server_params as search_server_params

# 简单配置类
class McpPolicyConfig(BaseModel):
    max_turns: int = 5
    max_tokens: int = 2048
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    mcp_server_name: str = "mcp_search"
    num_epochs: int = 4
    learning_rate: float = 1e-6

MCP_SERVERS = {
    "mcp_search": {
        "server_params": search_server_params,
        "val_data": "mcp_search/scenarios/val.jsonl"
    }
}

# 定义模型集合
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
