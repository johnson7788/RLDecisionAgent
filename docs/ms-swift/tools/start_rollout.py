#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/29 13:08
# @File  : start_rollout.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 启动 Qwen rollout 服务

from swift.llm import rollout_main, RolloutArguments

if __name__ == "__main__":
    args = RolloutArguments(
        # ==== 核心：Rollout ====
        model="Qwen/Qwen2.5-3B-Instruct",   # 选择的模型
        vllm_use_async_engine=True,          # 使用异步引擎
        # multi_turn_scheduler="tool_call_scheduler",  # 多轮调度器
        max_turns=5,                        # 最大回合数
    )

    rollout_main(args)
