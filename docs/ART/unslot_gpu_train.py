#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/19 21:18
# @File  : unslot_gpu_train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 单卡训练

import asyncio
from art import TrainableModel, gather_trajectory_groups
from art.local.backend import LocalBackend

# === 你的任务超简化：构造一条“回合”，并假装有打分 ===
async def toy_rollout_and_score(model: TrainableModel):
    # 让模型跑一小段“对话”（真实项目里这里是你的代理任务）
    msgs = [{"role":"user","content":"Say hello in one sentence."}]
    out = await model.complete(messages=msgs, max_tokens=32)
    # 伪造一个奖励（真实项目里：根据任务成败/指标来打分）
    reward = 1.0 if "hello" in out.lower() else -1.0
    # 封装为训练需要的“轨迹组”
    traj = [{"messages": msgs + [{"role":"assistant","content": out}]}]
    return gather_trajectory_groups([(traj, reward)])

async def main():
    backend = LocalBackend()  # 本机后端：vLLM + （默认）Unsloth 训练服务
    model = TrainableModel(
        name="agent-unsloth-1gpu",
        project="demo-unsloth",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",  # LoRA 单卡示例，模型随需替换
    )
    await model.register(backend)

    # 收集一小批“有分数的轨迹组”
    scored_groups = await toy_rollout_and_score(model)
    # === 用 Unsloth 在单卡上做一次训练步（LoRA） ===
    await model.train(scored_groups)

if __name__ == "__main__":
    asyncio.run(main())
