#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/19 21:20
# @File  : torchtune_gpu_train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 多卡进行训练,
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_torchtune_2gpu.py

import asyncio
from art import TrainableModel, gather_trajectory_groups
from art.local.backend import LocalBackend

# === 小任务与打分（同上） ===
async def toy_rollout_and_score(model: TrainableModel):
    msgs = [{"role":"user","content":"Write a short haiku about autumn."}]
    out = await model.complete(messages=msgs, max_tokens=32)
    reward = 1.0 if len(out.split()) >= 3 else -1.0
    traj = [{"messages": msgs + [{"role":"assistant","content": out}]}]
    return gather_trajectory_groups([(traj, reward)])

async def main():
    backend = LocalBackend()  # vLLM + torchtune（分布式）
    model = TrainableModel(
        name="agent-tune-2gpu",
        project="demo-torchtune",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",  # 例：做全参/部分全参微调
    )
    await model.register(backend)

    scored_groups = await toy_rollout_and_score(model)

    # === 关键：通过 torchtune 的分布式来多卡训练 ===
    # 说明：ART 集成的 torchtune service 会读取分布式上下文（torchrun 环境变量）
    # 并按分布式 full finetune 路径运行。下面的 torchtune_args 是给配方/训练器的透传入口；
    # 字段名取决于所用 torchtune 配方（如 full_finetune_distributed 等）。
    torchtune_args = {
        # 常见可调示例（根据你的显存和配方改）：
        "global_batch_size": 8,
        "micro_batch_size": 1,     # 每进程每步
        "grad_accum_steps": 4,
        # "use_fsdp": True,
        # "enable_activation_checkpointing": True,
        # "max_steps": 50,
    }

    await model.train(scored_groups, torchtune_args=torchtune_args)

if __name__ == "__main__":
    asyncio.run(main())
