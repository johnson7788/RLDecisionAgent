#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 08:06
# @File  : unsloth_thinking.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

# @Desc  : Qwen3 Thinking 训练（参考官方 notebook），复用 unsloth_core

from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from datasets import load_dataset, Dataset
from unsloth_core import (
    clip_display_text,
    TrainConfig,
    setup_logging,
    set_seed,
    log_env_info,
    build_model_and_tokenizer,
    build_trainer,
    train_and_report,
    save_model,
)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Qwen3-4B-Thinking SFT（unsloth_core 驱动）")

    # 与原脚本等价的关键默认值
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Thinking-2507")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.add_argument("--full_finetuning", action="store_true", default=False)
    parser.add_argument("--no_full_finetuning", dest="full_finetuning", action="store_false")
    parser.add_argument("--hf_token", type=str, default=None)

    # LoRA/训练超参（与原值一致）
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Thinking 专用 chat 模板
    parser.add_argument("--chat_template", type=str, default="qwen3-thinking")

    # 训练超参
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=3407)

    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_4b_thinking_lora")

    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        hf_token=args.hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        chat_template=args.chat_template,
        # dataset_name/split 由专用数据准备函数接管，这里不必设置
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=None,  # 以 max_steps 为准
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to="none",  # 默认不联动 W&B（如需可自行开启）
        output_dir=args.output_dir,
        save_steps=2,
    )
    # Thinking 模型同样使用 responses-only 的分隔符（与原始脚本一致）
    cfg.instruction_part = "<|im_start|>user\n"
    cfg.response_part = "<|im_start|>assistant\n"
    cfg.dataset_text_field = "text"
    return cfg

def prepare_dataset_openmath_thinking(tokenizer, logger: logging.Logger, *, split: str = "cot") -> Dataset:
    """Qwen Thinking 官方示例所用的 OpenMathReasoning-mini → conversations → text。"""
    logger.info("加载 OpenMathReasoning-mini（split=%s）…", split)
    ds = load_dataset("unsloth/OpenMathReasoning-mini", split=split)

    def generate_conversation(examples):
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for p, s in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": p},
                {"role": "assistant", "content": s},
            ])
        return {"conversations": conversations}

    ds = ds.map(generate_conversation, batched=True, desc="build_conversations")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    ds = ds.map(formatting_prompts_func, batched=True, desc="apply_chat_template")
    try:
        logger.info("样本校验: %s", clip_display_text(ds[0]["text"]))
    except Exception:
        pass
    return ds

def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or parse_args()
    logger = setup_logging(cfg.output_dir)

    logger.info("===== 训练配置 =====")
    logger.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    set_seed(cfg.seed, logger)
    log_env_info(logger)

    # 模型 & 数据
    model, tokenizer = build_model_and_tokenizer(cfg, logger)
    dataset = prepare_dataset_openmath_thinking(tokenizer, logger, split="cot")

    # Trainer & 训练
    trainer = build_trainer(model, tokenizer, dataset, cfg, logger)
    stats = train_and_report(trainer, logger)

    # 保存
    save_model(trainer, tokenizer, cfg.output_dir, logger)

    logger.info(f"{float(stats.metrics.get('train_runtime', 0.0)):.2f} 秒 used for training.")
    logger.info("🎉 Thinking 训练结束。")