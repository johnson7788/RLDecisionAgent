#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth + TRL 单文件训练脚本（可直接 python 运行）
- 将原始 Jupyter Notebook 代码整理为常规 Python 程序
- 支持命令行参数配置模型、LoRA、训练超参、保存路径
- 默认使用 unsloth/OpenMathReasoning-mini 数据集（split="cot"）
python train_unsloth_qwen_GRPO.py \
  --model_name unsloth/Qwen3-4B-Thinking-2507 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_steps 60 \
  --learning_rate 2e-4 \
  --output_dir lora_model

"""
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch

# ============ 日志 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_grpo_unsloth")

# ============ 依赖导入 ============
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
except Exception as e:
    logger.error("无法导入 unsloth，请先安装: pip install unsloth")
    raise

try:
    from transformers import TextStreamer
except Exception as e:
    logger.error("无法导入 transformers，请先安装（建议 4.55.4）: pip install transformers==4.55.4")
    raise

try:
    from trl import SFTTrainer, SFTConfig
except Exception as e:
    logger.error("无法导入 trl，请先安装: pip install trl")
    raise

try:
    from datasets import load_dataset
except Exception as e:
    logger.error("无法导入 datasets，请先安装: pip install datasets")
    raise


# ============ 配置 ============
@dataclass
class TrainConfig:
    # 模型与量化
    model_name: str = "unsloth/Qwen3-4B-Thinking-2507"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False  # 若想全参微调改为 True（需更大显存）

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = None  # 见下方 default_target_modules()
    use_gradient_checkpointing: str = "unsloth"  # True / False / "unsloth"
    use_rslora: bool = False
    loftq: bool = False

    # 数据集
    dataset_name: str = "unsloth/OpenMathReasoning-mini"
    dataset_split: str = "cot"
    chat_template: str = "qwen3-thinking"

    # 训练超参
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: Optional[int] = 60  # 置 None 则用 num_train_epochs
    num_train_epochs: Optional[float] = None
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    optim: str = "adamw_8bit"  # 需要 bitsandbytes
    seed: int = 3407

    # 保存与推理
    output_dir: str = "lora_model"
    do_infer_after_train: bool = True
    infer_prompt: str = "Solve (x + 2)^2 = 0."
    generation_max_new_tokens: int = 256

    # 其他
    bf16: bool = False  # 若显卡支持 BF16，可开启
    resume_from_checkpoint: bool = False


def default_target_modules():
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ============ 数据预处理函数 ============
def generate_conversation(examples):
    problems = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ])
    return {"conversations": conversations}


def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}


# ============ 训练主流程 ============
def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)

    if cfg.target_modules is None:
        cfg.target_modules = default_target_modules()

    logger.info("加载基础模型与分词器: %s", cfg.model_name)
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        full_finetuning=cfg.full_finetuning,
    )

    logger.info("应用 LoRA 适配器")
    model = FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
        loftq_config=None if not cfg.loftq else {},
    )

    logger.info("设置 Chat Template: %s", cfg.chat_template)
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)

    logger.info("加载数据集: %s (split=%s)", cfg.dataset_name, cfg.dataset_split)
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    logger.info("构造会话样本 -> 文本字段")
    dataset = dataset.map(generate_conversation, batched=True)
    dataset = dataset.map(lambda ex: formatting_prompts_func(ex, tokenizer), batched=True)

    # SFT 配置
    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_train_epochs if cfg.max_steps is None else None,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        bf16=cfg.bf16,
        report_to="none",
        output_dir=cfg.output_dir,
        remove_unused_columns=False,
    )

    logger.info("构建 SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_args,
    )

    # 只在 assistant 区间计算 loss
    logger.info("启用 'train_on_responses_only' 掩码")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 显卡信息
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s, 显存总量: %.2f GB", gpu.name, gpu.total_memory / 1024**3)
    else:
        logger.warning("未检测到 CUDA，将在 CPU 上训练（速度会很慢）。")

    logger.info("开始训练 ...")
    train_result = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_state()

    # 保存 LoRA 适配器与分词器
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info("保存 LoRA 到: %s", cfg.output_dir)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # 记录指标
    metrics_path = os.path.join(cfg.output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, ensure_ascii=False, indent=2)
    logger.info("训练完成，指标已保存到 %s", metrics_path)

    # 简单推理验证
    if cfg.do_infer_after_train:
        logger.info("训练后做一次简单推理验证 ...")
        messages = [{"role": "user", "content": cfg.infer_prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextStreamer(tokenizer, skip_prompt=False)
        _ = model.generate(
            **inputs,
            max_new_tokens=cfg.generation_max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            streamer=streamer,
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Unsloth + TRL 常规训练脚本")
    # 模型相关
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Thinking-2507")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", action="store_true", help="关闭 4bit 量化")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--full_finetuning", action="store_true", default=False)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--use_gradient_checkpointing", type=str, default="unsloth")
    parser.add_argument("--use_rslora", action="store_true", default=False)
    parser.add_argument("--loftq", action="store_true", default=False)

    # 数据
    parser.add_argument("--dataset_name", type=str, default="unsloth/OpenMathReasoning-mini")
    parser.add_argument("--dataset_split", type=str, default="cot")
    parser.add_argument("--chat_template", type=str, default="qwen3-thinking")

    # 训练超参
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--seed", type=int, default=3407)

    # 保存与推理
    parser.add_argument("--output_dir", type=str, default="lora_model")
    parser.add_argument("--do_infer_after_train", action="store_true", default=False)
    parser.add_argument("--infer_prompt", type=str, default="Solve (x + 2)^2 = 0.")
    parser.add_argument("--generation_max_new_tokens", type=int, default=256)

    # 其他
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)

    args = parser.parse_args()

    # 处理布尔开关
    if args.no_load_in_4bit:
        args.load_in_4bit = False

    cfg = TrainConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=None,  # 使用默认
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_rslora=args.use_rslora,
        loftq=args.loftq,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        chat_template=args.chat_template,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=None if (args.num_train_epochs is not None) else args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        optim=args.optim,
        seed=args.seed,
        output_dir=args.output_dir,
        do_infer_after_train=args.do_infer_after_train,
        infer_prompt=args.infer_prompt,
        generation_max_new_tokens=args.generation_max_new_tokens,
        bf16=args.bf16,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    return cfg


def main():
    cfg = parse_args()
    logger.info("配置: %s", json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    train(cfg)


if __name__ == "__main__":
    main()
