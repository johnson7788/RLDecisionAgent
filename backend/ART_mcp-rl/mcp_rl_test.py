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

from mcp_rl.rollout import McpScenario, rollout

load_dotenv()

MODEL_NAME = "mcp-14b-alpha-001"
PROJECT_NAME = "mcp_alphavantage"

async def main():
    model = art.TrainableModel(name=MODEL_NAME, project=PROJECT_NAME, base_model=None)
    backend = LocalBackend(in_process=True)
    await model.register(backend)

    # 这里假设你有 raw_val_scenarios
    raw_val_scenarios = [{"task": "示例验证任务"}] * 4
    val_scenarios = [McpScenario(task_description=s["task"]) for s in raw_val_scenarios]

    for i, scenario in enumerate(val_scenarios):
        print(f"\n测试： {i+1}: {scenario.task_description}")
        result = await rollout(model, scenario)
        print("模型输出:", result.messages()[-1]["content"] if result.messages() else "No response")

if __name__ == "__main__":
    asyncio.run(main())
