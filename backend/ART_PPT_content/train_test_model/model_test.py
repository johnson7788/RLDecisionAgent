#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/19 16:49
# @File  : mcp_rl_test.md.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试模型的训练效果

# test_mcp_rl.py
import asyncio
from dotenv import load_dotenv
import art
from art.local import LocalBackend
from .rollout import McpScenario, rollout
# 使用不同的mcp工具的配置
from experiments_config import MCP_SERVERS

load_dotenv()

MODEL_NAME = "ppt_agent_01"
PROJECT_NAME = "ppt_project_01"
mcp_configs = MCP_SERVERS["mcp_search"]
server_params = mcp_configs["server_params"]

async def main():
    model = art.TrainableModel(name=MODEL_NAME, project=PROJECT_NAME, base_model="Qwen/Qwen2.5-0.5B-Instruct")
    backend = LocalBackend(in_process=True)
    await model.register(backend)

    #准备的测试数据，准备了2条
    raw_val_scenarios = [{"task": "中国情趣用品市场经历了怎样的发展与观念变迁？"},  {"task": "一群退休大爷如何在大棚中追求爱情与体现自我价值？"}]
    val_scenarios = [McpScenario(task_description=s["task"], server_params=server_params) for s in raw_val_scenarios]

    for i, scenario in enumerate(val_scenarios):
        print(f"\n测试： {i+1}: {scenario.task_description}")
        result = await rollout(model, scenario)
        print("模型输出:", result.messages()[-1]["content"] if result.messages() else "No response")

if __name__ == "__main__":
    asyncio.run(main())
