#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/21 09:47
# @File  : test_email_search_agent.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_email_search_agent.py

功能：
- 载入训练同款模型/后端（与训练完全一致的推理路径）；
- 从 HF 取一条场景运行 rollout 并打印对话/工具调用/最终回答。

说明：
- 所有 await 均封装在函数里，通过 asyncio.run(...) 执行。
- 依赖 train_email_search_agent.py 中的核心实现（import 复用）。
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

# 复用训练程序中的核心定义与函数
from train_email_search_agent import (
    EmailScenario,
    load_training_scenarios,
    rollout,
    setup_model_and_backend,
    create_email_database,   # 若测试前未建库，可复用
    DB_PATH,
)

def ensure_env():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required.")
    if not os.path.exists(DB_PATH):
        print("Email DB not found, building now (this may take a while)...")
        create_email_database()


async def run_test(sample_index: int = 0):
    """异步测试：与训练同路径的 rollout 推理"""
    model = await setup_model_and_backend()
    scenarios = load_training_scenarios(split="train", limit=2, max_messages=1, shuffle=True, seed=123)

    if sample_index >= len(scenarios):
        sample_index = 0
    s = scenarios[sample_index]

    print("\n=== Test Scenario ===")
    print(f"ID: {s.id}")
    print(f"Q:  {s.question}")
    print(f"GT: {s.answer}")
    print(f"Ref IDs: {s.message_ids}")
    print(f"Inbox: {s.inbox_address} | Date: {s.query_date}")
    print("=" * 60)

    es = EmailScenario.model_validate({"step": 0, "scenario": s.model_dump()})
    traj = await rollout(model, es)

    print("\n=== Agent Trajectory ===")
    for msg in traj.messages():
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            print(f"[{role}] tool_calls: {tool_calls}")
        if content:
            preview = content if len(content) <= 200 else content[:200] + " ..."
            print(f"[{role}] {preview}\n")

    print("=" * 60)
    if getattr(traj, "final_answer", None):
        print(f"Final Answer: {traj.final_answer.answer}")
        print(f"Source IDs : {traj.final_answer.source_ids}")
    else:
        print("No final answer provided.")
    print(f"\nExpected Answer: {s.answer}")
    print(f"Expected IDs   : {s.message_ids}")
    print("\n🎉 Test finished.\n")


def main():
    parser = argparse.ArgumentParser(description="Test the trained ART email search agent.")
    parser.add_argument("--sample-index", type=int, default=0, help="Which scenario to test within the sampled set.")
    args = parser.parse_args()

    ensure_env()
    asyncio.run(run_test(sample_index=args.sample_index))


if __name__ == "__main__":
    main()
