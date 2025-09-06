#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块化 Unsloth SFT 训练脚本（可复用模板） + Weights & Biases 集成
----------------------------------------------------
新增能力：
1) 训练过程自动上报到 W&B（loss/learning_rate/steps 等由 TRL 内置上报）
2) 关键资源指标（GPU 显存峰值、训练耗时等）自定义 wandb.log
3) 训练异常捕获并上报到 W&B（alert + run.summary 标记失败，finish(exit_code=1)）
4) 训练完成自动标记 run 成功状态并可选上传最终模型为 artifact
5) 支持命令行开关（--use_wandb、--wandb_project、--wandb_run_name ...）

# .env中配置WANDB_BASE_URL和WANDB_API_KEY

依赖：
  - python-dotenv
  - wandb
  - unsloth >= 2024.XX
  - transformers, peft, trl, datasets, bitsandbytes (若使用 8bit/4bit)
  - torch, numpy

示例用法：
  pip install wandb
  wandb login  # 或设置环境变量 WANDB_API_KEY
  python unsloth_sft_wandb.py \
    --report_to wandb \
    --wandb_project unsloth-sft \
    --wandb_run_name qwen3-4b-lora \
    --output_dir ./outputs/qwen3_4b_lora

注意：默认启用 W&B（use_wandb=True）。如需禁用：--no_use_wandb。
"""
from __future__ import annotations
import dotenv
import os
import sys
import json
import time
import random
import logging
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from pathlib import Path
import numpy as np
import torch
dotenv.load_dotenv()

# ---------- 可选导入 wandb ----------
try:
    import wandb
    WANDB_AVAILABLE = True
    print(f"WANDB_AVAILABLE是可用的，已经安装了wandb")
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False
    print(f"WANDB_AVAILABLE不可用，没有安装wandb")

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
    report_to: str = "wandb"  # 改为默认同步到 wandb

    # W&B 相关
    use_wandb: bool = True
    wandb_project: Optional[str] = "unsloth-sft"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["unsloth", "sft", "qwen3"])
    wandb_dir: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: str = "train"
    wandb_mode: Optional[str] = None  # None/"online"/"offline"/"disabled"
    wandb_notes: Optional[str] = None
    wandb_log_model: bool | str = False  # True/False/"checkpoint"/"end"

    # 输出/保存
    output_dir: str = "./outputs/qwen3_4b_lora"
    save_steps: Optional[int] = 2  # 定期保存
    save_total_limit: Optional[int] = None
    logging_dir: Optional[str] = None  # 自定义日志目录


# ==============================
# 日志工具
# ==============================

def setup_logging(output_dir: str, logging_dir: Optional[str] = None) -> logging.Logger:
    """配置日志：控制台 + 文件。"""
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
# W&B 工具
# ==============================

def setup_wandb(cfg: TrainConfig, logger: logging.Logger):
    """初始化 wandb。返回 run 对象或 None。"""
    if not cfg.use_wandb:
        logger.info("已禁用 W&B。")
        return None
    if not WANDB_AVAILABLE:
        logger.warning("未检测到 wandb 包，已跳过 W&B 集成。pip install wandb")
        return None

    # 允许通过环境变量覆盖
    project = cfg.wandb_project or os.getenv("WANDB_PROJECT") or "unsloth-sft"
    entity = cfg.wandb_entity or os.getenv("WANDB_ENTITY")

    run = wandb.init(
        project=project,
        entity=entity,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        job_type=cfg.wandb_job_type,
        dir=cfg.wandb_dir,
        tags=cfg.wandb_tags or None,
        notes=cfg.wandb_notes,
        mode=cfg.wandb_mode,  # None 使用默认
        config=asdict(cfg),
        reinit=False,
    )

    # 记录当前脚本代码，便于复现
    try:
        wandb.run.log_code(root=str(Path(__file__).resolve().parent))
    except Exception:
        pass

    logger.info(f"已连接 W&B：project={project}, run={run.name}")
    return run


def wandb_log_metrics(metrics: dict):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics)


def wandb_on_error(e: Exception, logger: logging.Logger):
    if WANDB_AVAILABLE and wandb.run is not None:
        try:
            # 尝试发出告警（企业/团队版更友好），失败则降级为普通日志
            try:
                wandb.alert(title="Training crashed", text=str(e), level=wandb.AlertLevel.ERROR)
            except Exception:
                pass
            wandb.run.summary["status"] = "failed"
            wandb.log({"error/exception": str(e)})
            wandb.finish(exit_code=1)
            logger.error("已将异常上报至 W&B（status=failed）")
        except Exception as ee:
            logger.error(f"上报 W&B 异常失败: {ee}")


def wandb_on_success(extra_summary: dict | None = None, exit_code: int = 0):
    if WANDB_AVAILABLE and wandb.run is not None:
        if extra_summary:
            for k, v in extra_summary.items():
                wandb.run.summary[k] = v
        wandb.run.summary["status"] = "success"
        wandb.finish(exit_code=exit_code)


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
        sample_txt = dataset[0][cfg.dataset_text_field]
        logger.info(f"样本预览: {sample_txt}")
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
        report_to=cfg.report_to,  # → "wandb" 时将自动上报
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

    # 将梯度/权重变化发送到 W&B（可选）
    if WANDB_AVAILABLE and wandb.run is not None and cfg.use_wandb:
        try:
            wandb.watch(trainer.model, log="gradients", log_freq=max(1, cfg.logging_steps))
        except Exception:
            pass

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
        wandb_log_metrics({
            "env/gpu_name": gpu_stats.name,
            "env/gpu_mem_gb": max_mem,
            "memory/start_reserved_gb": start_reserved,
        })

    logger.info("开始训练…")
    trainer_stats = trainer.train()
    logger.info("训练完成。")

    used_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0.0
    used_for_lora = round(used_reserved - start_reserved, 3)
    used_pct = round(used_reserved / max_mem * 100, 3) if max_mem else 0.0
    lora_pct = round(used_for_lora / max_mem * 100, 3) if max_mem else 0.0

    rt = float(trainer_stats.metrics.get('train_runtime', 0.0))
    logger.info(f"训练耗时 {rt:.2f} 秒（约 {rt/60:.2f} 分钟）。")
    logger.info(f"峰值保留显存 = {used_reserved} GB；其中训练增量 = {used_for_lora} GB。")
    if max_mem:
        logger.info(f"显存占用峰值占比 = {used_pct}%；训练增量占比 = {lora_pct}%。")

    # 自定义指标上报到 W&B
    wandb_log_metrics({
        "memory/peak_reserved_gb": used_reserved,
        "memory/peak_reserved_pct": used_pct,
        "memory/lora_delta_gb": used_for_lora,
        "memory/lora_delta_pct": lora_pct,
        "time/train_runtime_sec": rt,
        "trainer/global_step": getattr(trainer.state, "global_step", 0),
    })

    return trainer_stats


# ==============================
# 保存模型
# ==============================

def save_model(trainer: SFTTrainer, tokenizer, output_dir: str, logger: logging.Logger, *, log_artifact: bool | str = False) -> None:
    """尝试使用TRL/Transformers 保存，并可选上传到 W&B Artifact。"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Unsloth 保存模型 Trainer.save_model")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存至: {output_dir}")

        # 可选：上传模型目录为 artifact
        if log_artifact and WANDB_AVAILABLE and wandb.run is not None:
            art = wandb.Artifact(
                name=f"{Path(output_dir).name}-{wandb.run.id}",
                type="model",
                metadata={"framework": "transformers", "task": "sft"},
            )
            art.add_dir(output_dir)
            aliases = ["latest"]
            if isinstance(log_artifact, str):
                aliases.append(log_artifact)
            wandb.log_artifact(art, aliases=aliases)
            logger.info("模型已作为 W&B Artifact 上传。")
    except Exception as ee:
        logger.error(f"保存失败: {ee}")
        raise

# ==============================
# 主流程
# ==============================

def parse_bool_flag(parser: argparse.ArgumentParser, true_flag: str, false_flag: str, default: bool):
    """同时支持 --flag / --no_flag 的布尔开关。返回存入 args 的目标名。"""
    dest = true_flag.replace("--", "").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action="store_true")
    group.add_argument(false_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})
    return dest


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Unsloth SFT 训练脚本（通用模板 + W&B）")

    # 仅列出常用项；其余请直接在 dataclass 默认值中改
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

    # 初始化 W&B（尽早建立 run，记录环境/配置）
    run = setup_wandb(cfg, logger)

    try:
        # 构建模型与 tokenizer
        model, tokenizer = build_model_and_tokenizer(cfg, logger)

        # 准备数据集
        dataset = load_and_prepare_dataset(cfg, tokenizer, logger)

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

    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")
        wandb_on_error(e, logger)
        raise

    logger.info("🎉 全流程结束。")


if __name__ == "__main__":
    main(parse_args())
