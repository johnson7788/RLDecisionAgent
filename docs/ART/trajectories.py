#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/12 22:09
# @File  : traj.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

# -*- coding: utf-8 -*-
"""
一个最小示例：构造三条轨迹 -> 用 RULER 打分 -> 打印输入与输出。
运行方式（项目根目录）：
  PYTHONPATH=src python scripts/ruler_score_group_demo.py
或在 Windows PowerShell：
  $env:PYTHONPATH="src"; python scripts/ruler_score_group_demo.py
"""
import os
import litellm
import logging
import asyncio
from typing import List, Dict, Any
import dotenv
# 如果 art 的 __init__ 已导出 TrajectoryGroup，你也可以直接 import art 并使用 art.TrajectoryGroup
from art.trajectories import Trajectory
import art  # 用于构造 TrajectoryGroup（ruler_score_group 内部也引用了 art.TrajectoryGroup）
# from art.rewards.ruler import ruler_score_group
from my_ruler import ruler_score_group
logging.basicConfig(level=logging.DEBUG)
litellm._turn_on_debug()
dotenv.load_dotenv()

def build_math_trajectories() -> List[Trajectory]:
    """
    构造三条对同一问题的不同回答轨迹：
    - t_good: 正确且简洁
    - t_bad: 错误答案
    - t_meh: 正确但啰嗦（应比 good 分低一点，但仍高于 bad）
    """
    sys = {
        "role": "system",
        "content": "You are a concise math assistant. Answer strictly and briefly.",
    }
    user = {"role": "user", "content": "What is 12 + 30?"}

    t_good = Trajectory(
        messages_and_choices=[
            sys,
            user,
            {"role": "assistant", "content": "42"},
        ],
        reward=0.0,
    )
    t_bad = Trajectory(
        messages_and_choices=[
            sys,
            user,
            {"role": "assistant", "content": "41"},
        ],
        reward=0.0,
    )
    t_meh = Trajectory(
        messages_and_choices=[
            sys,
            user,
            {
                "role": "assistant",
                "content": "12 + 30 = 42. By the way, 42 is also a famous number...",
            },
        ],
        reward=0.0,
    )
    return [t_good, t_bad, t_meh]


async def main() -> None:
    # 1) 构造一个 TrajectoryGroup
    trajectories = build_math_trajectories()
    group = art.TrajectoryGroup(trajectories)

    # 2) 用 RULER 进行相对打分（无需任何训练）
    # scored_group = await ruler_score_group(
    #     group,
    #     judge_model="openai/gpt-4o-mini",   # 可改为 "openai/o3"
    #     extra_litellm_params={"temperature": 0, "max_tokens": 500},
    #     swallow_exceptions=False,           # 示例中直接抛错，便于调试
    #     debug=True,                         # 打开可在控制台看到评审模型的 JSON 推理
    # )
    #
    scored_group = await ruler_score_group(
        group,
        judge_model="openai/deepseek-chat",   # 可改为 "openai/o3"
        extra_litellm_params={"temperature": 0, "max_tokens": 500, "api_base": "https://api.deepseek.com/v1", "api_key": os.getenv("DEEPSEEK_API_KEY")},
        swallow_exceptions=False,           # 示例中直接抛错，便于调试
        debug=True,                         # 打开可在控制台看到评审模型的 JSON 推理
    )

    # 3) 打印输入与输出（关键信息）
    if scored_group is None:
        print("RULER 评审失败（swallow_exceptions=True 时会返回 None）")
        return

    print("\n========== RULER 打分结果 ==========")
    for idx, traj in enumerate(scored_group.trajectories, start=1):
        print(f"\n--- Trajectory {idx} ---")
        # 输入消息（供你核对 RULER 看到的内容）
        for m in traj.messages():
            role = m.get("role", "")
            content = m.get("content", "")
            print(f"{role}: {content}")

        # 输出指标：RULER 会把独立奖励备份到 metrics['independent_reward']，
        # 并把 RULER 分数写到 metrics['ruler_score']，同时将 traj.reward 替换为该分数
        print("independent_reward:", traj.metrics.get("independent_reward"))
        print("ruler_score:", traj.metrics.get("ruler_score"))
        print("final reward (== ruler_score):", traj.reward)

        # 解释信息会以日志形式写入（见 ruler_score_group 的实现）
        expl = next(
            (log for log in traj.logs if log.startswith("RULER explanation:")), None
        )
        print("explanation log:", expl)

    # 可能出现的异常（例如部分轨迹生成报错时）会出现在 group.exceptions
    if scored_group.exceptions:
        print("\nExceptions:")
        for e in scored_group.exceptions:
            print(f"- {e.type}: {e.message}")


if __name__ == "__main__":
    asyncio.run(main())
