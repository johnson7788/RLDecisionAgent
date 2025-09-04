#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-Thinking.ipynb
基于 Unsloth + TRL 的指令微调训练脚本（适配 Qwen-3 / Llama / Gemma 等）
=================================================================
- 将原始 Jupyter Notebook 代码整理为可直接运行的 Python 程序。
- 主要功能：模型加载（4bit/8bit）、LoRA 注入、数据预处理（会话模板）、
  仅对助手回答计算损失（mask 用户输入）、训练、保存（LoRA/合并权重/GGUF）、
  以及可选的推理演示与显存统计。
- 全部注释与说明为中文，便于团队维护与二次开发。

依赖（建议在新环境中安装）：
-----------------------------------------------------------------
pip install unsloth
pip install "transformers==4.55.4"
pip install "datasets>=2.19.0"
pip install "trl>=0.9.6"
# 根据你的 CUDA 版本安装 PyTorch（以下是 CUDA 12.8 示例索引）
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128

快速开始：
-----------------------------------------------------------------
python train_unsloth_qwen3.py \
  --model_name "unsloth/Qwen3-4B-Thinking-2507" \
  --dataset_name "unsloth/OpenMathReasoning-mini" \
  --split "cot" \
  --max_steps 60 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --output_dir "./outputs/lora_qwen3" \
  --demo_prompt "解方程 (x+2)^2=0"

更多参数请见 --help。
"""

import os
import sys
import json
import time
import math
import argparse
import logging
import random
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

# Unsloth：快速加载 & 低显存训练
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only


# -----------------------------
# 通用工具
# -----------------------------
def set_seed(seed: int):
    """设置随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger():
    """简单日志器。"""
    logger = logging.getLogger("unsloth-train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# -----------------------------
# 参数
# -----------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Unsloth/TRL 指令微调训练脚本")
    # 模型 & 量化
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Thinking-2507",
                        help="基础模型名称（Hugging Face Hub 或本地路径）。")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="模型最大序列长度。")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="是否使用 4bit 量化加载。")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="是否使用 8bit 量化加载。")
    parser.add_argument("--full_finetuning", action="store_true", default=False, help="是否进行全参微调（默认关闭，使用 LoRA）。")
    parser.add_argument("--hf_token", type=str, default=None, help="如需加载门控模型，提供 HF token。")

    # LoRA 相关
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA 秩（越大表达力越强，显存占用也更高）。")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha。")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout。")
    parser.add_argument("--lora_bias", type=str, default="none", help="LoRA bias 策略：none/ all / lora_only。")
    parser.add_argument("--use_rslora", action="store_true", default=False, help="是否启用 Rank-stabilized LoRA。")
    parser.add_argument("--use_gradient_checkpointing", type=str, default="unsloth",
                        help='梯度检查点，可选：false/true/"unsloth"。长上下文建议 "unsloth"。')

    # 数据
    parser.add_argument("--dataset_name", type=str, default="unsloth/OpenMathReasoning-mini",
                        help="Hugging Face 数据集名称或本地路径。")
    parser.add_argument("--split", type=str, default="cot", help="数据集 split。")
    parser.add_argument("--chat_template", type=str, default="qwen3-thinking",
                        help="会话模板名称（如：qwen3-thinking, llama3, gemma3 等）。")

    # 训练
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每张卡的 batch size。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数。")
    parser.add_argument("--warmup_steps", type=int, default=5, help="学习率预热步数。")
    parser.add_argument("--num_train_epochs", type=float, default=0.0, help="训练轮数（>0 生效）。")
    parser.add_argument("--max_steps", type=int, default=60, help="训练步数（>0 生效，优先于 epoch）。")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率。长训建议 2e-5。")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减。")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="学习率调度器类型。")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="优化器（如 adamw_8bit/adamw_torch 等）。")
    parser.add_argument("--logging_steps", type=int, default=1, help="日志打印步数。")
    parser.add_argument("--report_to", type=str, default="none", help="日志上报平台（如 wandb/tensorboard/none）。")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子。")
    parser.add_argument("--mask_user_input", action="store_true", default=True,
                        help="仅对 assistant 的回答计算损失（忽略 user 部分）。")

    # 保存
    parser.add_argument("--output_dir", type=str, default="./outputs/lora_model", help="LoRA 适配器保存目录。")
    parser.add_argument("--save_merged16", action="store_true", help="是否另存为合并后的 16bit 权重（推理更简便）。")
    parser.add_argument("--save_merged4", action="store_true", help="是否另存为合并后的 4bit 权重。")
    parser.add_argument("--save_gguf", type=str, default="", help="保存为 GGUF 量化：q4_k_m/q5_k_m/q8_0/f16 等。空串不保存。")

    # Hub 推送（可选）
    parser.add_argument("--push_to_hub", action="store_true", help="是否推送到 HF Hub。需设置 HF token。")
    parser.add_argument("--hub_model_id", type=str, default="", help="HF Hub 目标仓库，如 yourname/your-model。")

    # 推理演示（可选）
    parser.add_argument("--demo_prompt", type=str, default="", help="若提供，将在训练后进行一次推理演示。")

    # 其他
    parser.add_argument("--print_sample", action="store_true", help="打印一个样例的模板化文本与 mask 后标签。")
    parser.add_argument("--show_memory", action="store_true", help="打印显存统计信息。")

    return parser


# -----------------------------
# 模型与数据准备
# -----------------------------
def load_model_and_tokenizer(args, logger):
    """加载基础模型与分词器，并注入 LoRA。"""
    logger.info(">>> 加载基础模型与分词器 ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        full_finetuning=bool(args.full_finetuning),
        token=args.hf_token,
    )

    logger.info(">>> 应用会话模板：%s", args.chat_template)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=args.chat_template,
    )

    # 仅当不是全参微调时，注入 LoRA 适配器
    if not args.full_finetuning:
        logger.info(">>> 注入 LoRA 适配器 ...")
        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            use_gradient_checkpointing=(
                True if str(args.use_gradient_checkpointing).lower() == "true"
                else ("unsloth" if str(args.use_gradient_checkpointing).lower() == "unsloth" else False)
            ),
            random_state=args.seed,
            use_rslora=bool(args.use_rslora),
            loftq_config=None,
        )
    else:
        logger.info(">>> 使用全参微调（未注入 LoRA）。")

    return model, tokenizer


def prepare_dataset(tokenizer, args, logger):
    """加载数据集，转为会话格式，并应用模板生成训练文本。"""
    logger.info(">>> 加载数据集：%s [%s]", args.dataset_name, args.split)
    dataset = load_dataset(args.dataset_name, split=args.split)

    # 将原始字段 problem/ generated_solution 转为多轮会话格式
    def generate_conversation(examples):
        problems = examples.get("problem", [])
        solutions = examples.get("generated_solution", [])
        conversations = []
        for p, s in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": p},
                {"role": "assistant", "content": s},
            ])
        return {"conversations": conversations}

    dataset = dataset.map(generate_conversation, batched=True, desc="构造会话数据")

    # 应用 chat template，生成纯文本字段 text
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, desc="应用模板到文本")
    logger.info(">>> 数据预处理完成，样本数：%d", len(dataset))

    return dataset


def build_trainer(model, tokenizer, dataset, args, logger):
    """构建 TRL 的 SFTTrainer。"""
    logger.info(">>> 构建 SFTTrainer ...")
    # 训练参数
    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=args.report_to,
        # 训练总步数 / 轮数（二选一）。若两者皆设置，HF 默认优先使用 max_steps。
        max_steps=int(args.max_steps) if args.max_steps and args.max_steps > 0 else None,
        num_train_epochs=float(args.num_train_epochs) if args.num_train_epochs and args.num_train_epochs > 0 else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,  # 如需评估，可传入验证集
        args=sft_args,
    )

    # 仅对助手回答计算损失（忽略 user 的 token）
    if args.mask_user_input:
        logger.info(">>> 启用 train_on_responses_only（仅对 assistant 的回答计算损失）")
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    return trainer


def print_one_sample(trainer, tokenizer, logger):
    """打印一个样本的模板化文本与 mask 后标签，便于核验。"""
    try:
        idx = 100 if len(trainer.train_dataset) > 100 else 0
        raw = tokenizer.decode(trainer.train_dataset[idx]["input_ids"])
        masked = tokenizer.decode([
            tokenizer.pad_token_id if x == -100 else x
            for x in trainer.train_dataset[idx]["labels"]
        ]).replace(tokenizer.pad_token, " ")
        logger.info(">>> 样本[%d] - 模板化文本：\n%s", idx, raw)
        logger.info(">>> 样本[%d] - Mask 后仅保留回答标签：\n%s", idx, masked)
    except Exception as e:
        logger.warning("打印样本失败：%s", e)


def show_memory_stats(prefix=""):
    """打印显存与设备信息（CUDA 可用时）。"""
    if not torch.cuda.is_available():
        print("[显存统计] 未检测到 CUDA。")
        return 0.0, 0.0, 0.0, 0.0, "CPU"
    gpu_props = torch.cuda.get_device_properties(0)
    start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_mem = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
    print(f"[显存统计] {prefix} GPU = {gpu_props.name} | Max = {max_mem} GB | 已保留 = {start_reserved} GB")
    return start_reserved, max_mem, 0.0, 0.0, gpu_props.name


# -----------------------------
# 训练 & 保存
# -----------------------------
def train_and_save(trainer, model, tokenizer, args, logger):
    """执行训练并保存 LoRA（及可选的合并权重 / GGUF）。"""
    # 训练前显存记录
    start_reserved, max_mem, _, _, gpu_name = show_memory_stats(prefix="开始")

    logger.info(">>> 开始训练 ...")
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    logger.info(">>> 训练完成，用时 %.2f 秒（约 %.2f 分钟）", train_time, train_time / 60.0)

    # 训练后显存记录
    if torch.cuda.is_available():
        used_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_for_lora = round(used_reserved - start_reserved, 3)
        used_pct = round(used_reserved / max_mem * 100, 3) if max_mem > 0 else 0.0
        lora_pct = round(used_for_lora / max_mem * 100, 3) if max_mem > 0 else 0.0
        logger.info(">>> 峰值显存保留 = %.3f GB，训练占用 = %.3f GB (%.3f%%)，总占用 = %.3f%%",
                    used_reserved, used_for_lora, lora_pct, used_pct)

    # 保存 LoRA 适配器
    logger.info(">>> 保存 LoRA 适配器到：%s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 可选：保存为合并后的 16bit / 4bit 权重（便于推理）
    # 注意：这会将 LoRA 与基础模型权重合并，生成独立可推理的权重目录。
    if args.save_merged16:
        merged16_dir = os.path.join(os.path.dirname(args.output_dir), "merged_16bit")
        os.makedirs(merged16_dir, exist_ok=True)
        logger.info(">>> 合并保存为 16bit 权重到：%s", merged16_dir)
        model.save_pretrained_merged(merged16_dir, tokenizer, save_method="merged_16bit")

    if args.save_merged4:
        merged4_dir = os.path.join(os.path.dirname(args.output_dir), "merged_4bit")
        os.makedirs(merged4_dir, exist_ok=True)
        logger.info(">>> 合并保存为 4bit 权重到：%s", merged4_dir)
        model.save_pretrained_merged(merged4_dir, tokenizer, save_method="merged_4bit")

    # 可选：保存为 GGUF（llama.cpp / Ollama 使用）
    if args.save_gguf:
        gguf_dir = os.path.join(os.path.dirname(args.output_dir), f"gguf_{args.save_gguf}")
        os.makedirs(gguf_dir, exist_ok=True)
        logger.info(">>> 保存 GGUF（%s）到：%s", args.save_gguf, gguf_dir)
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.save_gguf)

    # 可选：推送到 HF Hub
    if args.push_to_hub:
        if not args.hub_model_id:
            logger.warning("push_to_hub 已开启，但 hub_model_id 未设置，跳过推送。")
        else:
            logger.info(">>> 推送 LoRA 适配器到 HF Hub：%s", args.hub_model_id)
            model.push_to_hub(args.hub_model_id, token=args.hf_token)
            tokenizer.push_to_hub(args.hub_model_id, token=args.hf_token)

    # 返回训练指标（如需）
    metrics = getattr(train_result, "metrics", {})
    try:
        with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("保存训练指标失败：%s", e)

    return train_result


# -----------------------------
# 推理演示
# -----------------------------
def demo_inference(model, tokenizer, prompt: str, logger):
    """基于会话模板的推理演示（非“思维”模式）。"""
    logger.info(">>> 推理演示，用户问题：%s", prompt)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # 生成时必须
        enable_thinking=False,        # 关闭“思维链”模式
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 按 Qwen-3 官方建议的采样参数进行（可根据任务调整）
    _ = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7, top_p=0.8, top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    print()  # 末尾换行美观


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    logger = get_logger()
    set_seed(args.seed)

    # 1) 模型与 tokenizer
    model, tokenizer = load_model_and_tokenizer(args, logger)

    # 2) 数据准备
    dataset = prepare_dataset(tokenizer, args, logger)

    # 3) Trainer
    trainer = build_trainer(model, tokenizer, dataset, args, logger)

    # 4) 可选：打印一个样例，核对模板与 mask
    if args.print_sample:
        print_one_sample(trainer, tokenizer, logger)

    # 5) 训练 + 保存
    _ = train_and_save(trainer, model, tokenizer, args, logger)

    # 6) 可选：推理演示
    if args.demo_prompt:
        demo_inference(model, tokenizer, args.demo_prompt, logger)

    logger.info(">>> 全部流程结束。")


if __name__ == "__main__":
    main()
