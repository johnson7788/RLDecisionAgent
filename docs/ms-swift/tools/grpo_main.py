#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/28 21:17
# @File  : grpo_main.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

from swift.llm import rlhf_main, RLHFArguments

if __name__ == "__main__":
    args = RLHFArguments(
        # ==== 核心：RLHF/GRPO ====
        rlhf_type="grpo",
        model="Qwen/Qwen2.5-3B-Instruct",
        train_type="lora",
        lora_rank=8,
        lora_alpha=32,

        # ==== vLLM 推理后端（server 模式）====
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=["127.0.0.1"],
        vllm_server_port=[8000],

        # ==== 数据与长度 ====
        dataset=["zouxuhong/Countdown-Tasks-3to4#50000"],
        load_from_cache_file=True,
        max_length=2048,
        max_completion_length=1024,

        # ==== 训练超参 ====
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-7,
        gradient_accumulation_steps=8,
        warmup_ratio=0.01,

        # ==== 采样与GRPO相关 ====
        num_generations=8,
        temperature=1.0,
        beta=0.001,              # GRPO 的 β（KL/约束强度等），与你 CLI 的 --beta 相同
        num_iterations=1,        # 与 --num_iterations 一致

        # ==== 评测与保存 ====
        eval_steps=500,
        save_steps=100,
        save_total_limit=20,
        logging_steps=1,
        output_dir="output/GRPO_COUNTDOWN",

        # ==== 设备/数值精度/加速 ====
        torch_dtype="bfloat16",
        deepspeed="zero3",       # 内置会解析到对应的 ds_config/zero3.json

        # ==== 日志与可视化 ====
        log_completions=True,
        report_to=["tensorboard"],

        # ==== 系统提示词 ====
        system="You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",

        # ==== 自定义奖励与插件 ====
        # CLI 里的 "--reward_funcs external_countdown format" → Python 用列表写法：
        reward_funcs=["external_countdown", "format"],
        # CLI 里的 "--external_plugins examples/train/grpo/plugin/plugin.py"
        external_plugins=["examples/train/grpo/plugin/plugin.py"],
    )

    rlhf_main(args)
