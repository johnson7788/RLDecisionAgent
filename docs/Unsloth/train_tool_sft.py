#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script:
- logs to Weights & Biases (optional)
- authenticates to Hugging Face (optional)
- loads "unsloth/Meta-Llama-3.1-8B-Instruct"
- formats Salesforce/xlam-function-calling-60k into chat template tokens
- runs SFT with LoRA via Unsloth
- saves LoRA adapters and (optionally) a merged 16-bit checkpoint for vLLM

Usage example:
python train_tool_sft.py --wandb_project my-project --wandb_run llama31-fc --subset_size 15000 --epochs 3 --lr 2e-4 --batch_size 8 --grad_accum 2

Environment variables expected (if using these services):
- WANDB_API_KEY
- HUGGINGFACE_TOKEN

References:
- Unsloth docs: https://docs.unsloth.ai
- Medium article: https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060
"""
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import TrainingArguments
from trl import SFTTrainer

# Unsloth imports
from unsloth import FastLanguageModel, unsloth_train
from unsloth.chat_templates import get_chat_template

def maybe_login_wandb(project: str = None, run: str = None):
    """Initialize Weights & Biases logging if a project is provided."""
    if not project:
        print("[W&B] Skipping W&B init (no project provided).")
        return
    try:
        import wandb  # local import so it's optional
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise EnvironmentError("WANDB_API_KEY is not set in the environment variables.")
        wandb.login(key=api_key)
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_WATCH"] = "all"
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=project, name=run or "train_function_model")
        print(f"[W&B] Initialized: project={project} run={run}")
    except Exception as e:
        print(f"[W&B] Could not initialize W&B: {e}")


def maybe_login_hf():
    """Login to Hugging Face Hub if HUGGINGFACE_TOKEN is present."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        try:
            login(token)
            print("[HF] Logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"[HF] Login failed: {e}")
    else:
        print("[HF] HUGGINGFACE_TOKEN not set; skipping hub login.")


@dataclass
class TrainConfig:
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    dataset_name: str = "Salesforce/xlam-function-calling-60k"
    subset_size: int = 15000
    out_dir: str = "outputs"
    lora_dir: str = "lora_model"
    merged_dir: str = "merged_model_16bit"
    hf_username: str = ""
    push_to_hub: bool = False
    save_merged: bool = False

    # LoRA / Unsloth
    max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = False
    use_4bit: bool = False
    dtype: str = "auto"  # "auto" -> None for FastLanguageModel

    # Training
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 3
    lr: float = 2e-4
    warmup_steps: int = 5
    seed: int = 3407
    weight_decay: float = 0.01
    scheduler: str = "linear"
    optim: str = "adamw_8bit"  # requires bitsandbytes

    # Logging
    wandb_project: str = ""
    wandb_run: str = ""


def build_argparser():
    p = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B for function calling via LoRA (Unsloth).")
    p.add_argument("--model_name", default=TrainConfig.model_name)
    p.add_argument("--dataset_name", default=TrainConfig.dataset_name)
    p.add_argument("--subset_size", type=int, default=TrainConfig.subset_size)
    p.add_argument("--out_dir", default=TrainConfig.out_dir)
    p.add_argument("--lora_dir", default=TrainConfig.lora_dir)
    p.add_argument("--merged_dir", default=TrainConfig.merged_dir)
    p.add_argument("--hf_username", default=TrainConfig.hf_username)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--save_merged", action="store_true")

    p.add_argument("--max_seq_len", type=int, default=TrainConfig.max_seq_len)
    p.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    p.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    p.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)
    p.add_argument("--use_rslora", action="store_true")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--dtype", default=TrainConfig.dtype, help='Use "auto" for automatic dtype.')

    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--grad_accum", type=int, default=TrainConfig.grad_accum)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--scheduler", default=TrainConfig.scheduler)
    p.add_argument("--optim", default=TrainConfig.optim)

    p.add_argument("--wandb_project", default=TrainConfig.wandb_project)
    p.add_argument("--wandb_run", default=TrainConfig.wandb_run)
    return p


def format_dataset(tokenizer, dataset):
    # Use Unsloth chat template "llama-3" with ShareGPT-style mapping.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    def _format_batch(examples: Dict[str, List[Any]]):
        convos = []
        queries = examples.get("query") or examples.get("question") or []
        tools_list = examples.get("tools") or []
        answers = examples.get("answers") or examples.get("answer") or []

        for query, tools, ans in zip(queries, tools_list, answers):
            system_msg = {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to the following tools or function calls. "
                    "Your task is to produce a sequence of tools or function calls necessary to generate "
                    "response to the user utterance. Use the following tools or function calls as required:\n"
                    f"{tools}"
                ),
            }
            user_msg = {"role": "user", "content": str(query)}
            asst_msg = {"role": "assistant", "content": str(ans)}
            convos.append([system_msg, user_msg, asst_msg])

        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
        return {"text": texts}

    dataset = dataset.map(_format_batch, batched=True, desc="Formatting to chat template")
    return tokenizer, dataset


def main():
    args = build_argparser().parse_args()

    # Init logging/auth
    maybe_login_wandb(args.wandb_project, args.wandb_run)
    maybe_login_hf()

    torch.manual_seed(args.seed)

    # Load base model
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=dtype,
        load_in_4bit=args.use_4bit,
    )

    # Configure LoRA PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )

    # Load dataset
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    dataset = load_dataset(args.dataset_name, split="train", token=hf_token)
    if args.subset_size and args.subset_size > 0:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))
        print(f"[Data] Using subset of {len(dataset)} samples.")
    else:
        print(f"[Data] Using full dataset of {len(dataset)} samples.")

    # Tokenize/format
    tokenizer, dataset = format_dataset(tokenizer, dataset)

    # Training args
    train_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        output_dir=args.out_dir,
        report_to="wandb" if args.wandb_project else "none",
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="no",
        load_best_model_at_end=True,
        save_only_model=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=train_args,
    )

    # Memory stats (optional)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"[GPU] {gpu_stats.name}. Max memory = {max_mem} GB. Reserved at start = {start_gpu_mem} GB.")

    # Train
    stats = unsloth_train(trainer)
    print(stats)

    # Final memory
    if torch.cuda.is_available():
        used_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        start_gpu_mem = locals().get("start_gpu_mem", 0.0)
        used_for_lora = round(used_mem - start_gpu_mem, 3)
        gpu_stats = torch.cuda.get_device_properties(0)
        max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        used_pct = round(used_mem / max_mem * 100, 3)
        lora_pct = round(used_for_lora / max_mem * 100, 3)
        print(f"[Train] Runtime (s): {stats.metrics.get('train_runtime', 'NA')}")
        print(f"[Mem] Peak reserved = {used_mem} GB ({used_pct}%). For training = {used_for_lora} GB ({lora_pct}%).")

    # Save LoRA adapters
    os.makedirs(args.lora_dir, exist_ok=True)
    model.save_pretrained(args.lora_dir)
    tokenizer.save_pretrained(args.lora_dir)
    print(f"[Save] LoRA adapters saved to: {args.lora_dir}")

    # Push LoRA to hub
    if args.push_to_hub and args.hf_username:
        repo = f"{args.hf_username}/{os.path.basename(os.path.abspath(args.lora_dir))}"
        model.push_to_hub(repo, token=hf_token)
        tokenizer.push_to_hub(repo, token=hf_token)
        print(f"[Hub] Pushed LoRA adapters to: {repo}")

    # Optionally merge to a 16-bit full model for vLLM
    if args.save_merged:
        os.makedirs(args.merged_dir, exist_ok=True)
        model.save_pretrained_merged(args.merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[Save] Merged 16-bit model saved to: {args.merged_dir}")
        if args.push_to_hub and args.hf_username:
            repo = f"{args.hf_username}/{os.path.basename(os.path.abspath(args.merged_dir))}"
            model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", token=hf_token)
            print(f"[Hub] Pushed merged 16-bit model to: {repo}")

    # Close W&B run if any
    if args.wandb_project:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
