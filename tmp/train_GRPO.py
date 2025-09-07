#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO-style SFT training script for Unsloth + TRL with:
- argparse for CLI
- logging to console/file
- optional Weights & Biases tracking

Ref: https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb
"""

from __future__ import annotations
import os
import sys
import math
import json
import random
import logging
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset

# Optional: fail gracefully if wandb not installed
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# -----------------------------
# Utils
# -----------------------------

def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure Python logging.

    Args:
        level: Logging level name, e.g. "INFO", "DEBUG".
        log_file: Optional path to a log file. If provided, add a FileHandler.
    """
    # Basic config for root
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.captureWarnings(True)

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # speed on Ampere+


# -----------------------------
# Data formatting
# -----------------------------

def build_messages_row(
    x: pd.Series,
    system_prompt: str,
    reasoning_start: str,
    reasoning_end: str,
    solution_start: str,
    solution_end: str,
) -> List[Dict[str, str]]:
    expected_answer = str(x["expected_answer"])  # keep as string
    problem = x["problem"]
    thoughts = x.get("generated_solution", "") or ""

    # Remove any existing markers from the mined CoT
    if reasoning_start:
        thoughts = thoughts.replace(reasoning_start, "")
    if reasoning_end:
        thoughts = thoughts.replace(reasoning_end, "")

    thoughts = thoughts.strip()

    final_prompt = (
        f"{reasoning_start}{thoughts}{reasoning_end}"
        f"{solution_start}{expected_answer}{solution_end}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GRPO-style SFT with Unsloth + TRL"
    )

    # Model / LoRA
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Base")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization for base model")
    parser.add_argument("--fast_inference", type=lambda s: s.lower() != "false", default=True,
                        help="Enable Unsloth fast inference kernel (vLLM style)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Reasoning markers
    parser.add_argument("--reasoning_start", type=str, default="", help="Marker before reasoning (e.g., <think>)")
    parser.add_argument("--reasoning_end", type=str, default="", help="Marker after reasoning (e.g., </think>)")
    parser.add_argument("--solution_start", type=str, default="", help="Marker before final solution")
    parser.add_argument("--solution_end", type=str, default="", help="Marker after final solution")

    # Data
    parser.add_argument("--dataset_name", type=str, default="unsloth/OpenMathReasoning-mini")
    parser.add_argument("--dataset_split", type=str, default="cot")
    parser.add_argument("--max_prompt_ratio", type=float, default=0.5,
                        help="Keep samples whose tokenized length <= max_seq_length * ratio")

    # Training args
    parser.add_argument("--output_dir", type=str, default="outputs/unsloth-grpo")
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=[
                            "linear", "cosine", "cosine_with_restarts",
                            "polynomial", "constant", "constant_with_warmup",
                        ])
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        choices=["adamw_torch", "adamw_8bit"])  # bitsandbytes
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", help="Train in bf16 if available")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # W&B
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="grpo-unsloth")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="qwen3-4b-grpo")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)

    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_file", type=str, default=None)

    args = parser.parse_args()
    return args


def maybe_init_wandb(args: argparse.Namespace) -> bool:
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        return False

    if wandb is None:
        logging.warning("wandb is not installed; proceeding without experiment tracking.")
        os.environ["WANDB_DISABLED"] = "true"
        return False

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_MODE"] = args.wandb_mode  # online / offline

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config={k: v for k, v in vars(args).items() if "password" not in k.lower()},
        reinit=True,
        allow_val_change=True,
    )
    logging.info("Initialized Weights & Biases run: %s", run.name if run else "-")
    return True


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)

    logging.info("Arguments: %s", json.dumps(vars(args), indent=2, ensure_ascii=False))

    seed_everything(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # System prompt & chat_template similar to notebook, with placeholders filled after
    reasoning_start = args.reasoning_start
    reasoning_end = args.reasoning_end
    solution_start = args.solution_start
    solution_end = args.solution_end

    system_prompt = (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )
    logging.info("System prompt configured.")

    # Load base model
    logging.info("Loading base model: %s", args.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Apply LoRA
    logging.info("Configuring LoRA with rank=%d", args.lora_rank)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Build chat template with a placeholder + replace, to avoid Jinja escaping issues
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '{system_prompt}' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}{% endif %}"
    )
    # Replace placeholders safely (repr keeps quotes)
    chat_template = chat_template.replace("'{system_prompt}'", repr(system_prompt))
    chat_template = chat_template.replace("'{reasoning_start}'", repr(reasoning_start))
    tokenizer.chat_template = chat_template

    # Sanity-check template
    _ = tokenizer.apply_chat_template([
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role": "user", "content": "What is 2+2?"},
    ], tokenize=False, add_generation_prompt=True)

    # Load dataset
    logging.info("Loading dataset: %s (split=%s)", args.dataset_name, args.dataset_split)
    hf_ds = load_dataset(args.dataset_name, split=args.dataset_split)
    df = hf_ds.to_pandas()[["expected_answer", "problem", "generated_solution"]]

    # Filter to numeric answers only (as in the notebook)
    is_number = pd.to_numeric(pd.Series(df["expected_answer"]), errors="coerce").notnull()
    df = df.iloc[np.where(is_number)[0]].copy()
    logging.info("Dataset numeric-only size: %d", len(df))

    # Format to chat messages
    df["Messages"] = df.apply(
        lambda x: build_messages_row(
            x, system_prompt, reasoning_start, reasoning_end, solution_start, solution_end
        ),
        axis=1,
    )

    # Compute tokenized length and prune long ones
    df["N"] = df["Messages"].apply(lambda msgs: len(tokenizer.apply_chat_template(msgs)))
    max_tokens = int(args.max_seq_length * args.max_prompt_ratio)
    pruned = df.loc[df["N"] <= max_tokens].copy()
    logging.info(
        "Filtered dataset by token length: kept %d / %d (<= %d tokens)",
        len(pruned), len(df), max_tokens,
    )

    # Convert to HF Dataset with text field (string templated)
    pruned["text"] = tokenizer.apply_chat_template(pruned["Messages"].values.tolist(), tokenize=False)
    train_ds: Dataset = Dataset.from_pandas(pruned)

    # Maybe init wandb
    use_wandb = maybe_init_wandb(args)

    # Build Trainer
    bf16 = args.bf16 and torch.cuda.is_available()
    report_to = "wandb" if use_wandb else "none"

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=report_to,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=training_args,
    )

    # W&B model watching (optional; can be memory-heavy). Guarded.
    if use_wandb and wandb is not None:
        try:
            wandb.watch(model, log="gradients", log_freq=args.logging_steps)
        except Exception as e:  # pragma: no cover
            logging.warning("wandb.watch failed: %s", e)

    logging.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save artifacts
    logging.info("Saving model and tokenizer to: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)  # saves adapter if LoRA
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:  # pragma: no cover
        logging.warning("Tokenizer save failed: %s", e)

    if use_wandb and wandb is not None:
        wandb.finish()

    logging.info("All done ✅")


if __name__ == "__main__":
    main()
