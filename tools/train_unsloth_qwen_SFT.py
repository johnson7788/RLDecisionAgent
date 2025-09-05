#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/4 21:49
# @File  : train_unsloth_qwen_SFT.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 来自示例： https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-Instruct.ipynb
"""
## 主要功能

  * **命令行参数**：为常见配置（模型、LoRA、数据、训练、保存）提供命令行参数。
  * **可复现性**：确保种子可复现，并提供清晰的日志记录。
  * **数据格式化**：通过 Unsloth 聊天模板将数据格式化为 Qwen3 风格。
  * **训练掩码**：仅对助手的回复部分进行训练。
  * **简单推理检查**：包含一个简单的推理功能，用于确认训练效果。
  * **模型导出**：可保存 LoRA 适配器，并可选地导出合并后的 16 位、4 位或 GGUF 格式模型。

-----

## 安装要求

根据您的 CUDA 版本，运行以下命令安装所需库：

```bash
pip install unsloth
pip install transformers==4.55.4
pip install trl datasets
# PyTorch: 根据 https://pytorch.org/get-started/locally/ 选择正确的命令
# CUDA 12.8 示例：
# pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
```

-----

## 示例

```bash
# 基础训练示例
python train_unsloth_qwen_SFT.py \
    --model_name unsloth/Qwen3-4B-Instruct-2507 \
    --dataset_name mlabonne/FineTome-100k \
    --max_steps 60 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --save_dir ./lora_model

# 同时导出合并后的 16 位模型
python train_unsloth_qwen.py --merge_to_16bit --save_dir ./out_16bit
```

"""
from __future__ import annotations
import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def log_gpu_info(prefix: str = "") -> None:
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        reserved_gb = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
        logging.info(
            "%sGPU: %s | total=%.2f GB | reserved(max)=%.3f GB",
            prefix,
            props.name,
            props.total_memory / 1024 / 1024 / 1024,
            reserved_gb,
        )
    else:
        logging.info("%sRunning on CPU.", prefix)


# -----------------------------------------------------------------------------
# Core steps
# -----------------------------------------------------------------------------

@dataclass
class TrainArgs:
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    load_in_8bit: bool
    full_finetuning: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    use_gradient_checkpointing: str
    dataset_name: str
    dataset_split: str
    chat_template: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int
    num_train_epochs: float
    learning_rate: float
    logging_steps: int
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    mask_user_input: bool
    save_dir: str
    merge_to_16bit: bool
    merge_to_4bit: bool
    gguf_quant: Optional[str]
    push_to_hub_merged: Optional[str]
    push_to_hub_gguf: Optional[str]
    hf_token: Optional[str]
    inference_prompt: Optional[str]


def prepare_model_and_tokenizer(args: TrainArgs):
    logging.info("Loading base model: %s", args.model_name)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
    )

    # Apply LoRA
    logging.info("Attaching LoRA (r=%d, alpha=%d, dropout=%.3f).", args.lora_r, args.lora_alpha, args.lora_dropout)
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )

    # Chat template (Qwen3 format by default)
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    return model, tokenizer


def load_and_prepare_dataset(args: TrainArgs, tokenizer):
    logging.info("加载数据集: %s [%s]", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    logging.info("Standardizing dataset to conversation format if needed…")
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}

    logging.info("Applying chat template → text field…")
    dataset = dataset.map(formatting_prompts_func, batched=True, desc="templating")
    return dataset


def build_trainer(args: TrainArgs, model, tokenizer, dataset):
    # Build SFTTrainer
    sft_config_kwargs = dict(
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
        report_to="none",
    )

    if args.max_steps > 0:
        sft_config_kwargs.update(max_steps=args.max_steps)
    if args.num_train_epochs and args.num_train_epochs > 0:
        sft_config_kwargs.update(num_train_epochs=args.num_train_epochs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(**sft_config_kwargs),
    )

    if args.mask_user_input:
        logging.info("Masking user turns: training on assistant responses only…")
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    return trainer


def run_training(trainer: SFTTrainer) -> dict:
    logging.info("Starting training…")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_t = time.time()
    stats = trainer.train()
    dur = time.time() - start_t
    logging.info("Training complete in %.2f min (reported train_runtime=%.2fs).",
                 dur / 60, stats.metrics.get("train_runtime", float("nan")))
    return stats.metrics


def save_artifacts(args: TrainArgs, model, tokenizer) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("Saving LoRA adapters + tokenizer → %s", args.save_dir)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    if args.merge_to_16bit:
        out_dir = os.path.join(args.save_dir, "merged_fp16")
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Merging to 16-bit → %s", out_dir)
        model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
        if args.push_to_hub_merged:
            logging.info("Pushing merged 16-bit to Hub: %s", args.push_to_hub_merged)
            model.push_to_hub_merged(args.push_to_hub_merged, tokenizer, save_method="merged_16bit", token=args.hf_token)

    if args.merge_to_4bit:
        out_dir = os.path.join(args.save_dir, "merged_int4")
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Merging to 4-bit → %s", out_dir)
        model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_4bit")
        if args.push_to_hub_merged:
            logging.info("Pushing merged 4-bit to Hub: %s", args.push_to_hub_merged)
            model.push_to_hub_merged(args.push_to_hub_merged, tokenizer, save_method="merged_4bit", token=args.hf_token)

    if args.gguf_quant:
        q = args.gguf_quant
        out_dir = os.path.join(args.save_dir, f"gguf_{q}")
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Exporting GGUF (%s) → %s", q, out_dir)
        if q == "f16":
            model.save_pretrained_gguf(out_dir, tokenizer, quantization_method="f16")
        else:
            model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=q)
        if args.push_to_hub_gguf:
            logging.info("Pushing GGUF (%s) to Hub: %s", q, args.push_to_hub_gguf)
            model.push_to_hub_gguf(args.push_to_hub_gguf, tokenizer, quantization_method=q, token=args.hf_token)



def quick_inference(model, tokenizer, prompt: str, temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20, max_new_tokens: int = 128) -> None:
    if not prompt:
        return
    logging.info("Running a quick inference sample…")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune Qwen3 (or other) with Unsloth + TRL")

    # Model / LoRA
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--full_finetuning", action="store_true", default=False)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--use_rslora", action="store_true", default=False)
    p.add_argument("--use_gradient_checkpointing", type=str, default="unsloth", choices=["unsloth", "true", "false"])  # "true" treated same as True

    # Data
    p.add_argument("--dataset_name", type=str, default="mlabonne/FineTome-100k")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--chat_template", type=str, default="qwen3-instruct")

    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=60, help="Set >0 to use max_steps; set 0 to disable.")
    p.add_argument("--num_train_epochs", type=float, default=0.0, help="If >0, uses epochs instead of max_steps.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--optim", type=str, default="adamw_8bit")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--mask_user_input", action="store_true", default=True)

    # Saving / Export
    p.add_argument("--save_dir", type=str, default="./lora_model")
    p.add_argument("--merge_to_16bit", action="store_true", default=False)
    p.add_argument("--merge_to_4bit", action="store_true", default=False)
    p.add_argument("--gguf_quant", type=str, default=None, help="e.g. q4_k_m, q5_k_m, q8_0, f16")
    p.add_argument("--push_to_hub_merged", type=str, default=None, help="HF repo for merged upload, e.g. user/repo")
    p.add_argument("--push_to_hub_gguf", type=str, default=None, help="HF repo for gguf upload, e.g. user/repo")
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))

    # Inference sanity check
    p.add_argument("--inference_prompt", type=str, default="Continue the sequence: 1, 1, 2, 3, 5, 8,")

    # Misc
    p.add_argument("-v", "--verbose", action="count", default=1, help="-v (info), -vv (debug)")

    return p


def main():
    parser = build_arg_parser()
    ns = parser.parse_args()

    setup_logging(ns.verbose)
    set_seed(ns.seed)

    # Normalize gradient checkpointing flag
    if ns.use_gradient_checkpointing.lower() == "true":
        ns.use_gradient_checkpointing = True
    elif ns.use_gradient_checkpointing.lower() == "false":
        ns.use_gradient_checkpointing = False

    args = TrainArgs(
        model_name=ns.model_name,
        max_seq_length=ns.max_seq_length,
        load_in_4bit=ns.load_in_4bit,
        load_in_8bit=ns.load_in_8bit,
        full_finetuning=ns.full_finetuning,
        lora_r=ns.lora_r,
        lora_alpha=ns.lora_alpha,
        lora_dropout=ns.lora_dropout,
        use_rslora=ns.use_rslora,
        use_gradient_checkpointing=ns.use_gradient_checkpointing,
        dataset_name=ns.dataset_name,
        dataset_split=ns.dataset_split,
        chat_template=ns.chat_template,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        warmup_steps=ns.warmup_steps,
        max_steps=ns.max_steps,
        num_train_epochs=ns.num_train_epochs,
        learning_rate=ns.learning_rate,
        logging_steps=ns.logging_steps,
        optim=ns.optim,
        weight_decay=ns.weight_decay,
        lr_scheduler_type=ns.lr_scheduler_type,
        seed=ns.seed,
        mask_user_input=ns.mask_user_input,
        save_dir=ns.save_dir,
        merge_to_16bit=ns.merge_to_16bit,
        merge_to_4bit=ns.merge_to_4bit,
        gguf_quant=ns.gguf_quant,
        push_to_hub_merged=ns.push_to_hub_merged,
        push_to_hub_gguf=ns.push_to_hub_gguf,
        hf_token=ns.hf_token,
        inference_prompt=ns.inference_prompt,
    )

    # Prepare model + tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args)

    # Data
    dataset = load_and_prepare_dataset(args, tokenizer)

    # Trainer
    trainer = build_trainer(args, model, tokenizer, dataset)

    # Log GPU before train
    log_gpu_info(prefix="[Before] ")

    # Train
    metrics = run_training(trainer)

    # Log GPU after train
    log_gpu_info(prefix="[After ] ")

    # Pretty metrics summary
    if metrics:
        tr = metrics.get("train_runtime")
        if tr is not None:
            logging.info("train_runtime(s) = %.2f (%.2f min)", tr, tr/60)
        mmr = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0.0
        logging.info("peak_reserved_memory(GB) = %.3f", mmr)

    # Inference sample
    quick_inference(model, tokenizer, args.inference_prompt)

    # Save
    save_artifacts(args, model, tokenizer)

    logging.info("All done. Artifacts in: %s", args.save_dir)


if __name__ == "__main__":
    main()
