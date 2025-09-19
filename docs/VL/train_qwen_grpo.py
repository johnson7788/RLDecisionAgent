#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen_grpo.py
--------------------------------
用于 Qwen/Unsloth 模型的 GRPO 强化学习训练脚本。

说明：
- 依赖：unsloth、trl、transformers、datasets、accelerate、peft、torch
- 支持 Qwen2.5-VL（多模态）或纯文本模型。
- 奖励函数可选择内置（exact/substring/regex/nonempty）或通过 --reward_module 指定。
- 训练入口为 main()，可通过 argparse 配置参数。

示例：
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
from trl import GRPOTrainer, GRPOConfig  # 确保 trl 版本支持 GRPO
from unsloth import FastLanguageModel, FastVisionModel

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("训练脚本")

# -----------------------------
# 工具函数
# -----------------------------

def str2bool(v: str) -> bool:
    """将字符串转换为布尔值。

    支持的真值：yes/true/t/y/1（大小写不敏感）
    支持的假值：no/false/f/n/0（大小写不敏感）
    参数:
        v: 待转换的字符串或布尔值
    返回:
        bool: 转换后的布尔结果
    异常:
        ValueError: 当传入的字符串无法识别为布尔值时抛出
    """
    log.debug("str2bool 输入值 v=%r (type=%s)", v, type(v).__name__)
    if isinstance(v, bool):
        log.debug("str2bool 直接返回布尔值: %s", v)
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        log.debug("str2bool 解析为 True")
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        log.debug("str2bool 解析为 False")
        return False
    else:
        log.error("str2bool 遇到非法布尔字符串: %r", v)
        raise ValueError(f"Invalid boolean value: {v}")

def load_reward_module(path: str):
    """从给定路径加载自定义奖励函数模块。

    要求模块中必须包含函数：
        compute_reward(prompt, response, reference) -> float
    参数:
        path: 模块文件路径（.py）
    返回:
        callable: compute_reward 函数指针
    异常:
        ValueError: 当模块未提供 compute_reward 时抛出
    """
    log.info("加载自定义奖励模块: %s", path)
    spec = importlib.util.spec_from_file_location("reward_module", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_reward"):
        log.error("自定义奖励模块缺少 compute_reward 函数: %s", path)
        raise ValueError("Custom reward module must expose a function `compute_reward(prompt, response, reference) -> float`.")
    log.info("自定义奖励模块加载完成。")
    return mod.compute_reward

# -----------------------------
# 奖励函数定义
# -----------------------------

def reward_exact(prompt: str, response: str, reference: Optional[str]) -> float:
    """精确匹配奖励：标准化后，若 response 与 reference 完全相同给 1.0，否则 0.0。"""
    log.debug("reward_exact prompt(省略), response=%r, reference=%r", response, reference)
    if reference is None:
        return 0.0
    norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().casefold()
    score = 1.0 if norm(response) == norm(reference) else 0.0
    log.debug("reward_exact 得分: %s", score)
    return score

def reward_substring(prompt: str, response: str, reference: Optional[str]) -> float:
    """子串匹配奖励：若 reference 标准化后作为子串出现在 response 中则给 1.0，否则 0.0。"""
    log.debug("reward_substring response=%r, reference=%r", response, reference)
    if reference is None:
        return 0.0
    score = 1.0 if (reference or "").strip().casefold() in (response or "").casefold() else 0.0
    log.debug("reward_substring 得分: %s", score)
    return score

def reward_regex(prompt: str, response: str, reference: Optional[str]) -> float:
    """正则匹配奖励：将 reference 作为正则表达式，若可在 response 中匹配则给 1.0，否则 0.0。"""
    log.debug("reward_regex response=%r, reference=%r", response, reference)
    if reference is None:
        return 0.0
    try:
        score = 1.0 if re.search(reference, response or "", flags=re.IGNORECASE) else 0.0
        log.debug("reward_regex 得分: %s", score)
        return score
    except re.error as e:
        log.warning("参考答案包含无效正则: %r，错误: %s；回退到精确匹配。", reference, e)
        return reward_exact(prompt, response, reference)

def reward_nonempty(prompt: str, response: str, reference: Optional[str]) -> float:
    """非空奖励：若 response 非空（去除空白后）则给 1.0，否则 0.0。"""
    score = 1.0 if (response or "").strip() else 0.0
    log.debug("reward_nonempty response 非空=%s，得分=%s", bool((response or "").strip()), score)
    return score

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
    """使用 tokenizer 的 Chat 模板将用户消息包装为对话提示，并附加生成提示。"""
    log.debug("_as_chat_prompt user_text=%r", user_text[:100] + ("..." if len(user_text) > 100 else ""))
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},  # 追加生成提示
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    log.debug("_as_chat_prompt 生成的 prompt 长度=%d", len(prompt))
    return prompt

def _make_prompt(tokenizer, example: Dict[str, Any], use_chat_template: bool, user_field: str) -> str:
    """根据是否启用 Chat 模板，从样本中提取 user 字段并构造 prompt。"""
    log.debug("_make_prompt use_chat_template=%s, user_field=%s", use_chat_template, user_field)
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
    """加载并预处理数据集（HF Hub 或本地），并统一成训练需要的列。

    期望输入列：
      - user_field (默认 "prompt"): 用户输入文本
      - answer_field (默认 "answer"): 参考答案（大多奖励需要）
      - image_field  (默认 "image"): 图像路径或 PIL.Image（可选）
    参数:
      dataset_name_or_path: 数据集名称或本地路径
      tokenizer: 分词器/模板器
      split_train: 训练集切分名
      split_eval: 验证集切分名（可选）
      user_field/answer_field/image_field: 列名映射
      use_chat_template: 是否使用 Chat 模板包装
      max_train_samples/max_eval_samples: 限制样本数（便于快速调试）
    返回:
      (train_dataset, eval_dataset 或 None)
    """
    log.info("开始加载数据集: %s (train_split=%s, eval_split=%s)", dataset_name_or_path, split_train, split_eval)
    try:
        ds = load_dataset(dataset_name_or_path, data_files=None)
    except Exception as e:
        log.exception("数据集加载失败: %s", e)
        raise

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

    keep_cols_train = {user_field, answer_field, image_field}
    train_ds = train_ds.map(fmt, remove_columns=[c for c in train_ds.column_names if c not in keep_cols_train])
    if max_train_samples:
        original_len = len(train_ds)
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
        log.info("训练集裁剪: %d -> %d", original_len, len(train_ds))

    if eval_ds is not None:
        keep_cols_eval = {user_field, answer_field, image_field}
        eval_ds = eval_ds.map(fmt, remove_columns=[c for c in eval_ds.column_names if c not in keep_cols_eval])
        if max_eval_samples:
            original_len = len(eval_ds)
            eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
            log.info("验证集裁剪: %d -> %d", original_len, len(eval_ds))

    log.info("数据集加载完成：train=%d, eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "None")
    return train_ds, eval_ds

# -----------------------------
# 文本生成与奖励包装
# -----------------------------

def build_compute_rewards_fn(reward_type: str, reward_module_path: Optional[str] = None):
    """构建用于 GRPOTrainer 的奖励计算封装函数。

    优先使用自定义模块（若提供路径），否则使用内置奖励类型。
    参数:
        reward_type: 内置奖励类型名称（exact/substring/regex/nonempty）
        reward_module_path: 自定义奖励模块路径（可选）
    返回:
        callable: (samples, responses) -> List[float] 的函数
    """
    log.info("构建奖励函数: reward_type=%s, reward_module_path=%s", reward_type, reward_module_path)
    if reward_module_path:
        compute_reward = load_reward_module(reward_module_path)
    else:
        if reward_type not in REWARD_MAP:
            log.error("未知的 reward_type: %s，可选项: %s", reward_type, list(REWARD_MAP))
            raise ValueError(f"Unknown reward_type: {reward_type}. Choose from {list(REWARD_MAP)} or provide --reward_module.")
        compute_reward = REWARD_MAP[reward_type]

    def _fn(samples: List[Dict[str, Any]], responses: List[str]) -> List[float]:
        """对一批样本与模型输出计算奖励列表。"""
        rewards: List[float] = []
        for ex, resp in zip(samples, responses):
            r = float(compute_reward(ex.get("prompt", ""), resp, ex.get("reference")))
            rewards.append(r)
        log.debug("批次奖励统计: n=%d, 均值=%.4f, 最小=%.4f, 最大=%.4f",
                  len(rewards), (sum(rewards) / max(len(rewards), 1)) if rewards else 0.0,
                  min(rewards) if rewards else 0.0,
                  max(rewards) if rewards else 0.0)
        return rewards

    log.info("奖励函数构建完成。")
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
    """加载 Unsloth 快速模型，并根据配置注入 LoRA/PEFT。

    参数详见函数签名。
    返回:
        (model, tokenizer)
    """
    log.info(
        "加载模型: name=%s, vision=%s, 4bit=%s, max_seq_length=%d, LoRA(r=%d, alpha=%d, dropout=%.2f), "
        "tune(viz=%s, lang=%s, attn=%s, mlp=%s), gpu_mem_util=%.2f",
        model_name, vision, load_in_4bit, max_seq_length, lora_r, lora_alpha, lora_dropout,
        finetune_vision_layers, finetune_language_layers, finetune_attention_modules, finetune_mlp_modules,
        gpu_memory_utilization
    )

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
            target_modules=None, # 交由 Unsloth 自动挑选
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
    log.info("模型与分词器加载完成。")
    return model, tokenizer

# -----------------------------
# 训练逻辑
# -----------------------------

def train(args):
    """主训练流程：加载模型与数据、构建奖励、启动 GRPO 训练并保存结果。"""
    log.info("设置随机种子: %d", args.seed)
    set_seed(args.seed)
    vision = bool(args.vision)

    # 打印关键训练参数摘要
    log.info(
        "训练参数摘要: model=%s, dataset=%s, vision=%s, output_dir=%s, lr=%g, batch_per_device=%d, "
        "max_prompt_len=%d, max_completion_len=%d, episodes=%d, bf16=%s, fp16=%s, "
        "reward_type=%s, reward_module=%s",
        args.model_name, args.dataset, vision, args.output_dir, args.learning_rate, args.per_device_train_batch_size,
        args.max_prompt_length, args.max_completion_length, args.num_episodes, args.bf16, args.fp16,
        args.reward_type, args.reward_module
    )

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
    log.info("数据集就绪：训练样本=%d, 验证样本=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "None")

    # Rewards
    compute_rewards = build_compute_rewards_fn(args.reward_type, args.reward_module)

    # GRPO config (DR-GRPO 选项)
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
        loss_type=args.loss_type,  # 例如 "dr_grpo"
        importance_sampling_level=args.importance_sampling_level,  # "sequence" | "token"
        mask_truncated_completions=args.mask_truncated_completions,
    )
    log.info(
        "GRPO 配置: logging_steps=%d, save_steps=%d, grad_accu=%d, max_grad_norm=%.3f, scheduler=%s, optim=%s",
        args.logging_steps, args.save_steps, args.gradient_accumulation_steps, args.max_grad_norm,
        args.lr_scheduler_type, args.optim
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_rewards],  # 可扩展多奖励
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    log.info("开始训练...")
    trainer.train()
    log.info("训练完成，正在保存模型到: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("模型与分词器保存完成。")

    if args.push_to_hub:
        repo_id = args.hub_repo_id or os.path.basename(args.output_dir)
        log.info("推送模型到 HuggingFace Hub: repo_id=%s", repo_id)
        trainer.push_to_hub(repo_id)
        log.info("推送完成。")

# -----------------------------
# 命令行接口
# -----------------------------

def build_arg_parser():
    """构建命令行参数解析器，提供数据、模型、训练、奖励与杂项配置项。"""
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
    """命令行入口：解析参数、创建输出目录并启动训练。"""
    parser = build_arg_parser()
    args = parser.parse_args()
    log.info("命令行参数解析完成。输出目录: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()
