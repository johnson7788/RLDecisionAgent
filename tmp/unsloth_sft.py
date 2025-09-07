#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 08:05
# @File  : unsloth_sft.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
"""
模块化 Unsloth SFT 训练脚本（复用 unsloth_core） + 可选 W&B
与原版相比：
- 训练主体逻辑复用 unsloth_core
- 保留原有 CLI 风格（常用参数）
"""
from __future__ import annotations
import argparse
import json
import dotenv
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
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
dotenv.load_dotenv()


def parse_bool_flag(parser: argparse.ArgumentParser, true_flag: str, false_flag: str, default: bool):
    dest = true_flag.replace("--", "").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action="store_true")
    group.add_argument(false_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})
    return dest


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Unsloth SFT 训练（unsloth_core 驱动）")

    # 常用项（默认值来自 TrainConfig）
    parser.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--max_seq_length", type=int, default=TrainConfig.max_seq_length)
    parse_bool_flag(parser, "--load_in_4bit", "--no_load_in_4bit", default=TrainConfig.load_in_4bit)
    parse_bool_flag(parser, "--load_in_8bit", "--no_load_in_8bit", default=TrainConfig.load_in_8bit)
    parse_bool_flag(parser, "--full_finetuning", "--no_full_finetuning", default=TrainConfig.full_finetuning)
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)

    parser.add_argument("--chat_template", type=str, default=TrainConfig.chat_template)
    parser.add_argument("--dataset_name", type=str, default=TrainConfig.dataset_name)
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

    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--save_steps", type=int, default=TrainConfig.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=None)

    # W&B
    parse_bool_flag(parser, "--use_wandb", "--no_use_wandb", default=False)
    parser.add_argument("--wandb_project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=TrainConfig.wandb_entity)
    parser.add_argument("--wandb_run_name", type=str, default=TrainConfig.wandb_run_name)
    parser.add_argument("--wandb_group", type=str, default=TrainConfig.wandb_group)
    parser.add_argument("--wandb_job_type", type=str, default=TrainConfig.wandb_job_type)
    parser.add_argument("--wandb_mode", type=str, default=TrainConfig.wandb_mode)
    parser.add_argument("--wandb_dir", type=str, default=TrainConfig.wandb_dir)
    parser.add_argument("--wandb_notes", type=str, default=TrainConfig.wandb_notes)
    parser.add_argument("--wandb_log_model", type=str, default=str(TrainConfig.wandb_log_model))
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)

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
        wandb_tags=args.wandb_tags,
        wandb_dir=args.wandb_dir,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_mode=args.wandb_mode,
        wandb_notes=args.wandb_notes,
        wandb_log_model=wandb_log_model,
    )
    return cfg

def prepare_dataset_generic(cfg: TrainConfig, tokenizer, logger: logging.Logger) -> Dataset:
    """通用：标准化到 conversations → 渲染为 cfg.dataset_text_field。"""
    logger.info(f"加载数据集: {cfg.dataset_name} [{cfg.dataset_split}] …")
    raw_ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    preview_n = max(1, min(3, len(raw_ds)))
    logger.info(f"原始样本预览（前 {preview_n} 条）:")
    for i in range(preview_n):
        try:
            logger.info(f"[#%d] 原始(raw): %s", i, _pjson(raw_ds[i]))
        except Exception as e:
            logger.warning(f"[#%d] 原始样本读取失败: %s", i, e)

    logger.info("标准化为 conversations …")
    std_ds = standardize_data_formats(raw_ds)

    logger.info("渲染为文本字段…")
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {cfg.dataset_text_field: texts}

    dataset = std_ds.map(formatting_prompts_func, batched=True, desc="apply_chat_template")

    try:
        logger.info("渲染样本校验: %s", clip_display_text(dataset[0][cfg.dataset_text_field]))
    except Exception:
        pass
    return dataset


def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()

    logger = setup_logging(cfg.output_dir)
    logger.info("===== 训练配置 =====")
    logger.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    set_seed(cfg.seed, logger)
    log_env_info(logger)

    run = setup_wandb(cfg, logger)

    try:
        model, tokenizer = build_model_and_tokenizer(cfg, logger)
        dataset = prepare_dataset_generic(cfg, tokenizer, logger)
        trainer = build_trainer(model, tokenizer, dataset, cfg, logger)
        stats = train_and_report(trainer, logger)
        save_model(trainer, tokenizer, cfg.output_dir, logger, log_artifact=(cfg.wandb_log_model or False))
        extra = {"metrics/train_runtime_sec": float(stats.metrics.get("train_runtime", 0.0))}
        wandb_on_success(extra_summary=extra, exit_code=0)
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")
        wandb_on_error(e, logger)
        raise

    logger.info("🎉 全流程结束。")


if __name__ == "__main__":
    main(parse_args())