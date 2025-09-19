#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen_grpo.py (中文注释版)
--------------------------------
用于 Qwen/Unsloth 模型的 GRPO 强化学习训练脚本。

说明：
- 依赖：unsloth、trl、transformers、datasets、accelerate、peft、torch
- 支持 Qwen2.5-VL（多模态）或纯文本模型。
- 奖励函数可选择内置（exact/substring/regex/nonempty）或通过 --reward_module 指定。
- 训练入口为 main()，可通过 argparse 配置参数。

python train_qwen_grpo.py \
  --dataset your_dataset_on_hub_or_local \
  --model_name unsloth/Qwen2.5-VL-7B-Instruct \
  --vision true \
  --user_field prompt --answer_field answer --image_field image \
  --output_dir outputs_qwen_vl_grpo
"""

import os
import re
import json
import math
import random
import logging
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, set_seed
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig  # Make sure trl version supports GRPO
from unsloth import FastLanguageModel, FastVisionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("训练脚本")

# -----------------------------
# 工具函数
# -----------------------------

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {v}")

def load_reward_module(path: str):
    spec = importlib.util.spec_from_file_location("reward_module", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_reward"):
        raise ValueError("Custom reward module must expose a function `compute_reward(prompt, response, reference) -> float`.")
    return mod.compute_reward

# -----------------------------
# 奖励函数定义
# -----------------------------

def reward_exact(prompt: str, response: str, reference: Optional[str]) -> float:
    """1.0 if normalized response==reference else 0.0."""
    if reference is None:
        return 0.0
    norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().casefold()
    return 1.0 if norm(response) == norm(reference) else 0.0

def reward_substring(prompt: str, response: str, reference: Optional[str]) -> float:
    """1.0 if reference substring appears in response, else 0."""
    if reference is None:
        return 0.0
    return 1.0 if (reference or "").strip().casefold() in (response or "").casefold() else 0.0

def reward_regex(prompt: str, response: str, reference: Optional[str]) -> float:
    """
    Treat `reference` as a regular expression and grant 1.0 if it matches the response, else 0.0.
    Use with care; escape your patterns if you mean literal matches.
    """
    if reference is None:
        return 0.0
    try:
        return 1.0 if re.search(reference, response or "", flags=re.IGNORECASE) else 0.0
    except re.error:
        log.warning("参考答案中包含无效的正则表达式，回退到精确匹配。")
        return reward_exact(prompt, response, reference)

def reward_nonempty(prompt: str, response: str, reference: Optional[str]) -> float:
    """Give 1.0 if response is non-empty; useful as a sanity check baseline."""
    return 1.0 if (response or "").strip() else 0.0

REWARD_MAP = {
    "exact": reward_exact,
    "substring": reward_substring,
    "regex": reward_regex,
    "nonempty": reward_nonempty,
}

# -----------------------------
# 数据集处理
# -----------------------------

def _as_chat_prompt(tokenizer, user_text: str) -> str:
    """Wrap a single user message using the tokenizer's chat template."""
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},  # add generation prompt
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _make_prompt(tokenizer, example: Dict[str, Any], use_chat_template: bool, user_field: str) -> str:
    if use_chat_template:
        return _as_chat_prompt(tokenizer, example[user_field])
    else:
        return example[user_field]

def load_and_prepare_dataset(
    dataset_name_or_path: str,
    tokenizer,
    split_train: str = "train",
    split_eval: Optional[str] = None,
    user_field: str = "prompt",
    answer_field: str = "answer",
    image_field: Optional[str] = "image",
    use_chat_template: bool = True,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load a dataset from HF hub or local files. Expects fields:
      - user_field (default: "prompt") : str
      - answer_field (default: "answer"): str (reference answer) -- optional but needed for most rewards
      - image_field (default: "image")  : path or PIL.Image (optional)
    Returns (train_dataset, eval_dataset)
    """
    if os.path.exists(dataset_name_or_path) and os.path.isdir(dataset_name_or_path):
        data_files = None
    else:
        data_files = None

    ds = load_dataset(dataset_name_or_path, data_files=data_files)
    if isinstance(ds, DatasetDict):
        train_ds = ds[split_train]
        eval_ds = ds[split_eval] if (split_eval and split_eval in ds) else None
    else:
        train_ds, eval_ds = ds, None

    def fmt(example):
        prompt = _make_prompt(tokenizer, example, use_chat_template, user_field)
        out = {
            "prompt": prompt,
            "reference": example.get(answer_field, None),
        }
        if image_field and image_field in example:
            out["image"] = example[image_field]
        return out

    train_ds = train_ds.map(fmt, remove_columns=[c for c in train_ds.column_names if c not in {user_field, answer_field, image_field}])
    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if eval_ds is not None:
        eval_ds = eval_ds.map(fmt, remove_columns=[c for c in eval_ds.column_names if c not in {user_field, answer_field, image_field}])
        if max_eval_samples:
            eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds

# -----------------------------
# 文本生成与奖励包装
# -----------------------------

def build_compute_rewards_fn(reward_type: str, reward_module_path: Optional[str] = None):
    if reward_module_path:
        compute_reward = load_reward_module(reward_module_path)
    else:
        if reward_type not in REWARD_MAP:
            raise ValueError(f"Unknown reward_type: {reward_type}. Choose from {list(REWARD_MAP)} or provide --reward_module.")
        compute_reward = REWARD_MAP[reward_type]

    def _fn(samples: List[Dict[str, Any]], responses: List[str]) -> List[float]:
        rewards: List[float] = []
        for ex, resp in zip(samples, responses):
            rewards.append(float(compute_reward(ex.get("prompt", ""), resp, ex.get("reference"))))
        return rewards
    return _fn

# -----------------------------
# 模型加载
# -----------------------------

def load_unsloth_model(
    model_name: str,
    vision: bool,
    load_in_4bit: bool,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    finetune_vision_layers: bool,
    finetune_language_layers: bool,
    finetune_attention_modules: bool,
    finetune_mlp_modules: bool,
    gpu_memory_utilization: float = 0.8,
):
    if vision:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=finetune_vision_layers,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=None, # Let Unsloth pick
            use_rslora=False,
            loftq_config=None,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=None,
            use_rslora=False,
            loftq_config=None,
        )
    return model, tokenizer

# -----------------------------
# 训练逻辑
# -----------------------------

def train(
    args,
):
    set_seed(args.seed)
    vision = bool(args.vision)

    # Load model + tokenizer
    model, tokenizer = load_unsloth_model(
        model_name=args.model_name,
        vision=vision,
        load_in_4bit=args.load_in_4bit,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Datasets
    train_ds, eval_ds = load_and_prepare_dataset(
        dataset_name_or_path=args.dataset,
        tokenizer=tokenizer,
        split_train=args.train_split,
        split_eval=args.eval_split,
        user_field=args.user_field,
        answer_field=args.answer_field,
        image_field=args.image_field if vision else None,
        use_chat_template=not args.no_chat_template,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    # Rewards
    compute_rewards = build_compute_rewards_fn(args.reward_type, args.reward_module)

    # GRPO config (DR-GRPO options exposed as args where useful)
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_episodes=args.num_episodes,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm,
        report_to=None if not args.report_to else args.report_to,
        loss_type=args.loss_type,  # e.g., "dr_grpo"
        importance_sampling_level=args.importance_sampling_level,  # "sequence" | "token"
        mask_truncated_completions=args.mask_truncated_completions,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_rewards],  # you can add more for multi-reward
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    log.info("开始训练...")
    trainer.train()
    log.info("训练完成，正在保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        log.info("正在推送模型到 HuggingFace Hub...")
        trainer.push_to_hub(args.hub_repo_id or os.path.basename(args.output_dir))

# -----------------------------
# 命令行接口
# -----------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="用于 Qwen/Unsloth 模型的 GRPO 强化学习训练。支持文本与多模态。")

    # 数据相关
    p.add_argument("--dataset", type=str, required=True, help="数据集名称（HF Hub）或本地路径。")
    p.add_argument("--train_split", type=str, default="train", help="训练集分片名称（默认 train）。")
    p.add_argument("--eval_split", type=str, default=None, help="验证集分片名称（默认无）。")
    p.add_argument("--user_field", type=str, default="prompt", help="用户输入字段名。")
    p.add_argument("--answer_field", type=str, default="answer", help="参考答案字段名。")
    p.add_argument("--image_field", type=str, default="image", help="图像字段名（多模态训练时使用）。")
    p.add_argument("--max_train_samples", type=int, default=None, help="训练样本最大数量（默认使用全部）。")
    p.add_argument("--max_eval_samples", type=int, default=None, help="验证样本最大数量（默认使用全部）。")
    p.add_argument("--no_chat_template", action="store_true", help="禁用 Chat 模板包装。")

    # 模型相关
    p.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-VL-7B-Instruct", help="模型名称或路径。")
    p.add_argument("--vision", type=str2bool, default=True, help="是否使用多模态模型（True=Qwen2.5-VL，False=纯文本）。")
    p.add_argument("--load_in_4bit", type=str2bool, default=True, help="是否使用 4bit 量化加载模型。")
    p.add_argument("--max_seq_length", type=int, default=16384, help="最大序列长度。")

    # LoRA 相关
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank 参数。")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha 参数。")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout 概率。")
    p.add_argument("--finetune_vision_layers", type=str2bool, default=False, help="是否微调视觉层。")
    p.add_argument("--finetune_language_layers", type=str2bool, default=True, help="是否微调语言层。")
    p.add_argument("--finetune_attention_modules", type=str2bool, default=True, help="是否微调 Attention 模块。")
    p.add_argument("--finetune_mlp_modules", type=str2bool, default=True, help="是否微调 MLP 模块。")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="显存利用率上限（0~1）。")

    # 训练相关
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录。")
    p.add_argument("--learning_rate", type=float, default=5e-6, help="学习率。")
    p.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减。")
    p.add_argument("--warmup_steps", type=int, default=50, help="学习率预热步数。")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型。")
    p.add_argument("--optim", type=str, default="adamw_torch", help="优化器类型。")
    p.add_argument("--logging_steps", type=int, default=5, help="日志打印间隔步数。")
    p.add_argument("--save_steps", type=int, default=50, help="保存模型的间隔步数。")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数。")
    p.add_argument("--per_device_train_batch_size", type=int, default=1, help="每张显卡上的 batch size。")
    p.add_argument("--max_prompt_length", type=int, default=1024, help="输入提示的最大长度。")
    p.add_argument("--max_completion_length", type=int, default=512, help="生成结果的最大长度。")
    p.add_argument("--num_episodes", type=int, default=1, help="每个样本的 GRPO rollout 数量。")
    p.add_argument("--bf16", type=str2bool, default=True, help="是否启用 bfloat16 精度。")
    p.add_argument("--fp16", type=str2bool, default=False, help="是否启用 float16 精度。")
    p.add_argument("--max_grad_norm", type=float, default=0.1, help="最大梯度裁剪值。")
    p.add_argument("--report_to", type=str, default=None, help="日志上报工具，例如 'wandb'。")
    p.add_argument("--loss_type", type=str, default="dr_grpo", help="GRPO 损失函数类型，例如 'dr_grpo'。")
    p.add_argument("--importance_sampling_level", type=str, default="sequence", choices=["sequence", "token"], help="重要性采样级别：序列/标记。")
    p.add_argument("--mask_truncated_completions", type=str2bool, default=False, help="是否屏蔽被截断的生成结果。")

    # 奖励相关
    p.add_argument("--reward_type", type=str, default="exact", choices=list(REWARD_MAP.keys()),
                   help="奖励函数类型（内置：exact|substring|regex|nonempty）。")
    p.add_argument("--reward_module", type=str, default=None,
                   help="自定义奖励函数模块路径（需包含 compute_reward(prompt, response, reference) 方法）。")

    # 其它
    p.add_argument("--seed", type=int, default=42, help="随机种子。")
    p.add_argument("--push_to_hub", type=str2bool, default=False, help="是否推送到 HuggingFace Hub。")
    p.add_argument("--hub_repo_id", type=str, default=None, help="推送到 Hub 时使用的仓库 ID。")

    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()
