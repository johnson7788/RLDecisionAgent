#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块化 Unsloth SFT 训练脚本（可复用模板）
----------------------------------------------------
将原始脚本封装为通用训练模块：
1) 函数化结构 + 配置化参数（dataclass/CLI）
2) 中文注释，便于团队维护
3) 统一日志（控制台 + 文件），记录训练全流程
4) 支持 4bit/8bit/LoRA 与 Qwen3 聊天模板示例
5) 数据集标准化 + ChatTemplate 格式化 + 仅训练 Assistant 响应

依赖：
  - unsloth >= 2024.XX
  - transformers, peft, trl, datasets, bitsandbytes (若使用 8bit/4bit)
  - torch, numpy

示例用法：
python unsloth_sft.py

注意：默认使用 Qwen3 指令模板（qwen3-instruct），并且只训练 assistant 段落。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np
import torch

# Unsloth & 训练相关库
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig


# ==============================
# 配置定义
# ==============================
@dataclass
class TrainConfig:
    # 模型 / Tokenizer
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    hf_token: Optional[str] = None  # 若模型为 gated，可在此传递 token

    # LoRA / PEFT 设置
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"  # True / False / "unsloth"
    use_rslora: bool = False
    loftq_config: Optional[dict] = None
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Chat 模板/数据处理
    chat_template: str = "qwen3-instruct"
    dataset_name: str = "mlabonne/FineTome-100k"
    dataset_split: str = "train"
    dataset_text_field: str = "text"  # 生成后的文本字段名

    # 仅训练 assistant 响应的分隔符（依据模板）：
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"

    # 训练超参
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60  # 或使用 num_train_epochs
    num_train_epochs: Optional[int] = 4
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"  # 需 bitsandbytes
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"  # 可切换为 wandb

    # 输出/保存
    output_dir: str = "./outputs/qwen3_4b_lora"
    save_steps: Optional[int] = 2  # 定期保存
    save_total_limit: Optional[int] = None
    logging_dir: Optional[str] = None  # 自定义日志目录


# ==============================
# 日志工具
# ==============================

def setup_logging(output_dir: str, logging_dir: Optional[str] = None) -> logging.Logger:
    """配置日志：控制台 + 文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = logging_dir or os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, time.strftime("train_%Y%m%d_%H%M%S.log"))

    logger = logging.getLogger("unsloth_sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(ch_formatter)

    # 文件输出
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"日志写入: {log_file}")
    return logger


# ==============================
# 通用工具
# ==============================

def set_seed(seed: int, logger: logging.Logger | None = None) -> None:
    """设置随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 更快训练；如需完全确定性可设 True
    torch.backends.cudnn.benchmark = True
    if logger:
        logger.info(f"随机种子已设置: {seed}")


def log_env_info(logger: logging.Logger) -> None:
    """记录环境与 GPU 信息。"""
    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        max_mem = round(gpu.total_memory / 1024 / 1024 / 1024, 3)
        start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu.name}  显存上限: {max_mem} GB  启动保留: {start_reserved} GB")
    else:
        logger.warning("未检测到可用 CUDA，训练将使用 CPU（可能非常慢）")


# ==============================
# 构建模型 & Tokenizer
# ==============================

def build_model_and_tokenizer(cfg: TrainConfig, logger: logging.Logger):
    """加载基础模型，并注入 LoRA 适配器；应用 Chat 模板。"""
    if cfg.load_in_4bit and cfg.load_in_8bit:
        logger.warning("4bit 与 8bit 仅能二选一，已优先使用 4bit，并关闭 8bit！")
        cfg.load_in_8bit = False

    logger.info("开始加载基础模型…")
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        full_finetuning=cfg.full_finetuning,
        token=cfg.hf_token,
    )
    logger.info("基础模型加载完成。")

    logger.info("注入 LoRA 适配器…")
    model = FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
        loftq_config=cfg.loftq_config,
    )
    logger.info("LoRA 适配器注入完成。")

    logger.info(f"应用 Chat 模板: {cfg.chat_template}")
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    return model, tokenizer


# ==============================
# 数据集准备
# ==============================

def load_and_prepare_dataset(cfg: TrainConfig, tokenizer, logger: logging.Logger) -> Dataset:
    """加载数据集，统一对话格式，并转成训练字段 text。"""
    logger.info(f"加载数据集: {cfg.dataset_name} [{cfg.dataset_split}] …")
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    logger.info("标准化为通用 conversations 格式…")
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        # 将多轮对话用 ChatTemplate 渲染为纯文本（含特殊标记）
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {cfg.dataset_text_field: texts}

    logger.info("根据 Chat 模板渲染文本字段…")
    dataset = dataset.map(formatting_prompts_func, batched=True, desc="apply_chat_template")

    # 打印/记录一个样本便于检查
    try:
        sample_txt = dataset[0][cfg.dataset_text_field][:256].replace("\n", " ")
        logger.info(f"样本预览: {sample_txt}…")
    except Exception as e:
        logger.warning(f"样本预览失败: {e}")

    return dataset


# ==============================
# 构建 Trainer
# ==============================

def build_trainer(model, tokenizer, dataset: Dataset, cfg: TrainConfig, logger: logging.Logger) -> SFTTrainer:
    """创建 SFTTrainer，并设置仅训练回答段落。"""
    logger.info("创建 SFTTrainer…")

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        dataset_text_field=cfg.dataset_text_field,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        # 也可选择设定 num_train_epochs（两者不要同时强制）
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        report_to=cfg.report_to,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
    )

    logger.info("切换为仅学习 assistant 响应（忽略 user 指令部分的 loss）…")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=cfg.instruction_part,
        response_part=cfg.response_part,
    )

    # 记录一个编码后的样本，确认 mask 是否合理（只做日志预览，不影响训练）
    try:
        decoded = tokenizer.decode(trainer.train_dataset[0]["input_ids"][:256])
        logger.info(f"编码样本预览: {decoded}")
    except Exception as e:
        logger.warning(f"编码样本预览失败: {e}")

    return trainer


# ==============================
# 训练与度量
# ==============================

def train_and_report(trainer: SFTTrainer, logger: logging.Logger):
    """启动训练并记录显存与耗时。"""
    gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0.0
    max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3) if gpu_stats else 0.0

    if gpu_stats:
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_mem} GB.")
        logger.info(f"启动时保留显存 = {start_reserved} GB.")

    logger.info("开始训练…")
    trainer_stats = trainer.train()
    logger.info("训练完成。")

    used_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0.0
    used_for_lora = round(used_reserved - start_reserved, 3)
    used_pct = round(used_reserved / max_mem * 100, 3) if max_mem else 0.0
    lora_pct = round(used_for_lora / max_mem * 100, 3) if max_mem else 0.0

    rt = trainer_stats.metrics.get('train_runtime', 0.0)
    logger.info(f"训练耗时 {rt:.2f} 秒（约 {rt/60:.2f} 分钟）。")
    logger.info(f"峰值保留显存 = {used_reserved} GB；其中训练增量 = {used_for_lora} GB。")
    if max_mem:
        logger.info(f"显存占用峰值占比 = {used_pct}%；训练增量占比 = {lora_pct}%。")

    return trainer_stats


# ==============================
# 保存模型
# ==============================

def save_model(trainer: SFTTrainer, tokenizer, output_dir: str, logger: logging.Logger) -> None:
    """尝试使用TRL/Transformers 保存。"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Unsloth 保存模型 Trainer.save_model")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存至: {output_dir}")
    except Exception as ee:
        logger.error(f"保存失败: {ee}")
        raise

# ==============================
# 主流程
# ==============================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Unsloth SFT 训练脚本（通用模板）")

    # 仅列出常用项；其余请直接在 dataclass 默认值中改
    parser.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--max_seq_length", type=int, default=TrainConfig.max_seq_length)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--full_finetuning", action="store_true", default=False)
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

    args = parser.parse_args()

    # 将 argparse 合并到 dataclass（未暴露的字段沿用默认值）
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
    )
    return cfg


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

    # 构建模型与 tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, logger)

    # 准备数据集
    dataset = load_and_prepare_dataset(cfg, tokenizer, logger)

    # 构建 Trainer
    trainer = build_trainer(model, tokenizer, dataset, cfg, logger)

    # 训练并报告
    try:
        stats = train_and_report(trainer, logger)
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")
        raise

    # 保存模型
    try:
        save_model(trainer, tokenizer, cfg.output_dir, logger)
    except Exception as e:
        logger.exception(f"保存模型失败: {e}")
        raise

    logger.info("🎉 全流程结束。")


if __name__ == "__main__":
    main(parse_args())
