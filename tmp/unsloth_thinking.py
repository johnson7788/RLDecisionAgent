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
import dotenv
import logging
from dataclasses import asdict
from datasets import load_dataset, Dataset
from unsloth_core import (
    clip_display_text,
    TrainConfig,
    setup_logging,
    set_seed,
    log_env_info,
    setup_wandb,
    wandb_on_error,
    wandb_on_success,
    build_model_and_tokenizer,
    build_trainer,
    train_and_report,
    save_model,
)
dotenv.load_dotenv()


def parse_bool_flag(parser: argparse.ArgumentParser, true_flag: str, false_flag: str, default: bool):
    """同时支持 --flag / --no_flag 的布尔开关。返回存入 args 的目标名。"""
    dest = true_flag.replace("--", "").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action="store_true")
    group.add_argument(false_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})
    return dest


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Thinking SFT（unsloth_core 驱动）")

    # 与原脚本等价的关键默认值
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Thinking-2507")
    parser.add_argument("--max_seq_length", type=int, default=TrainConfig.max_seq_length)
    parse_bool_flag(parser, "--load_in_4bit", "--no_load_in_4bit", default=TrainConfig.load_in_4bit)
    parse_bool_flag(parser, "--load_in_8bit", "--no_load_in_8bit", default=TrainConfig.load_in_8bit)
    parse_bool_flag(parser, "--full_finetuning", "--no_full_finetuning", default=TrainConfig.full_finetuning)
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)

    # Thinking 专用 chat 模板
    parser.add_argument("--chat_template", type=str, default="qwen3-thinking")
    parser.add_argument("--dataset_name", type=str, default="unsloth/OpenMathReasoning-mini")
    parser.add_argument("--dataset_split", type=str, default=TrainConfig.dataset_split)

    parser.add_argument("--per_device_train_batch_size", type=int, default=TrainConfig.per_device_train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--num_train_epochs", type=float, default=TrainConfig.num_train_epochs)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--logging_steps", type=int, default=TrainConfig.logging_steps)
    parser.add_argument("--optim", type=str, default=TrainConfig.optim)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--lr_scheduler_type", type=str, default=TrainConfig.lr_scheduler_type)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--report_to", type=str, default=TrainConfig.report_to)

    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_4b_thinking_lora")
    parser.add_argument("--save_steps", type=int, default=TrainConfig.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=None)

    # W&B 相关开关
    parse_bool_flag(parser, "--use_wandb", "--no_use_wandb", default=TrainConfig.use_wandb)
    parser.add_argument("--wandb_project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=TrainConfig.wandb_entity)
    parser.add_argument("--wandb_run_name", type=str, default=TrainConfig.wandb_run_name)
    parser.add_argument("--wandb_group", type=str, default=TrainConfig.wandb_group)
    parser.add_argument("--wandb_job_type", type=str, default=TrainConfig.wandb_job_type)
    parser.add_argument("--wandb_mode", type=str, default=TrainConfig.wandb_mode)
    parser.add_argument("--wandb_dir", type=str, default=TrainConfig.wandb_dir)
    parser.add_argument("--wandb_notes", type=str, default=TrainConfig.wandb_notes)
    parser.add_argument("--wandb_log_model", type=str, default=str(TrainConfig.wandb_log_model))
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="空格分隔的 tag 列表")

    args = parser.parse_args()

    # 解析 wandb_log_model 为 bool/str
    wandb_log_model: bool | str
    if str(args.wandb_log_model).lower() in {"true", "1", "yes"}:
        wandb_log_model = True
    elif str(args.wandb_log_model).lower() in {"false", "0", "no"}:
        wandb_log_model = False
    else:
        wandb_log_model = str(args.wandb_log_model)

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
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=args.report_to,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags or TrainConfig().wandb_tags,
        wandb_dir=args.wandb_dir,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_mode=args.wandb_mode,
        wandb_notes=args.wandb_notes,
        wandb_log_model=wandb_log_model,
    )
    # Thinking 模型同样使用 responses-only 的分隔符（与原始脚本一致）
    cfg.instruction_part = "<|im_start|>user\n"
    cfg.response_part = "<|im_start|>assistant\n"
    cfg.dataset_text_field = "text"
    return cfg

def prepare_dataset_openmath_thinking(cfg: TrainConfig, tokenizer, logger: logging.Logger, *, split: str = "cot") -> Dataset:
    """Qwen Thinking 官方示例所用的 OpenMathReasoning-mini → conversations → text。"""
    logger.info(f"加载数据集: {cfg.dataset_name} [{cfg.dataset_split}] …")
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

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
    cfg = cfg or TrainConfig()

    # 日志
    logger = setup_logging(cfg.output_dir, cfg.logging_dir)

    # 打印配置
    logger.info("===== 训练配置 =====")
    logger.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    # 随机种子 & 环境信息
    set_seed(cfg.seed, logger)
    log_env_info(logger)

    # 初始化 W&B（尽早建立 run，记录环境/配置）
    run = setup_wandb(cfg, logger)

    # 构建模型与 tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, logger)

    # 准备数据集
    dataset = prepare_dataset_openmath_thinking(cfg, tokenizer, logger, split="cot")

    # 构建 Trainer
    trainer = build_trainer(model, tokenizer, dataset, cfg, logger)

    # 训练并报告
    stats = train_and_report(trainer, logger)

    # 保存模型 & 可选上传 artifact（别名：final）
    save_model(
        trainer,
        tokenizer,
        cfg.output_dir,
        logger,
        log_artifact=(cfg.wandb_log_model if cfg.wandb_log_model else False),
    )

    # 成功收尾
    extra = {"metrics/train_runtime_sec": float(stats.metrics.get("train_runtime", 0.0))}
    wandb_on_success(extra_summary=extra, exit_code=0)

    logger.info(f"{float(stats.metrics.get('train_runtime', 0.0)):.2f} 秒 used for training.")
    logger.info("🎉  训练结束。")

if __name__ == "__main__":
    main(parse_args())
