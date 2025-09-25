#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen_vl_grpo.py — 仅视觉 GRPO 训练脚本

主要特点:
- 仅视觉：多模态流。
- 与 Unsloth GRPO/GSPO notebook 中的 VLM 语义保持一致。
- 针对 MathVista (AI4Math/MathVista, split=testmini) 提供安全默认值。
- 两阶段奖励：格式 + 正确性，使用 <REASONING>/<SOLUTION> 标签。
- 正确的多模态对话模板，带图像占位符和 tokenizer 处理。
- 使用 TRL 的 GRPOTrainer (>=0.22.x 语义) 和 Unsloth FastVisionModel。

使用示例:不要使用./unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit,因为使用fast_inference=True了
--model_name ./unsloth/Qwen2.5-VL-3B-Instruct # 使用本地模型
--model_name unsloth/Qwen2.5-VL-3B-Instruct  #使用huggingface的cache的模型
python train_qwen_grpo.py \
  --dataset AI4Math/MathVista \
  --train_split testmini \
  --model_name ./unsloth/Qwen2.5-VL-3B-Instruct \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --output_dir outputs_qwen_vl_grpo --fast_inference no --load_in_4bit no
"""

import os
import re
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset, DatasetDict
from unsloth import FastVisionModel
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("vl_grpo")

# ====== 特殊标记：用于奖励函数和提示词格式化 ======
REASONING_START = "<REASONING>"
REASONING_END   = "</REASONING>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

# ====== 奖励函数 ======

def formatting_reward_func(completions, **kwargs):
    """奖励规则：
    - 如果输出中包含且仅包含一个推理块，则加 1 分。
    - 如果输出中包含且仅包含一个答案块，则再加 1 分。
    - 如果输出中存在 addCriterion 或换行符异常堆叠，则扣 2 分。
    """
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern   = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    scores = []
    for completion in completions:
        score = 0.0
        t = re.findall(thinking_pattern, completion, re.DOTALL)
        a = re.findall(answer_pattern,   completion, re.DOTALL)
        if len(t) == 1: score += 1.0
        if len(a) == 1: score += 1.0
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0
        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """奖励规则：
    - 如果 <SOLUTION> 标签内仅包含一个浮点数，且与参考答案完全一致，则得 2 分。
    - 否则得 0 分。
    """
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    responses = [re.findall(answer_pattern, c, re.DOTALL) for c in completions]
    scores = []
    for r, a in zip(responses, answer):
        ok = (len(r) == 1) and (a == r[0].replace("\n", ""))
        scores.append(2.0 if ok else 0.0)
    return scores

# ====== 数据预处理 ======

def _vision_prompt_dict(question: str) -> List[Dict[str, Any]]:
    """构造多模态对话字典：图像 + 文本提示。"""
    text_content = (
        f"{question}。请先在 {REASONING_START} 与 {REASONING_END} 之间写下推理过程，"
        f"然后在 {SOLUTION_START} 与 {SOLUTION_END} 之间写下最终答案（必须是一个浮点数）。"
    )
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text_content},
        ]},
    ]


def _resize_rgb(ex, image_field: str) -> Dict[str, Any]:
    """将图像调整为 512x512 并转为 RGB。"""
    img = ex.get("decoded_image", ex.get(image_field))
    if img is None:
        return ex
    img = img.resize((512, 512))
    if img.mode != "RGB":
        img = img.convert("RGB")
    ex["image"] = img
    return ex


def load_and_prepare_dataset(
    dataset_name_or_path: str,
    tokenizer,
    split_train: str = "testmini",
    split_eval: Optional[str] = None,
    user_field: str = "question",
    answer_field: str = "answer",
    image_field: str = "decoded_image",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """加载数据集，过滤非数字答案，处理图像，构造提示，并返回训练/验证集。"""
    logger.info("加载数据集: %s", dataset_name_or_path)
    ds = load_dataset(dataset_name_or_path)

    if isinstance(ds, DatasetDict):
        if split_train not in ds:
            raise ValueError(f"数据集没有 split '{split_train}'。可选: {list(ds.keys())}")
        train_ds = ds[split_train]
        eval_ds = ds[split_eval] if (split_eval and split_eval in ds) else None
    else:
        train_ds, eval_ds = ds, None

    # 保留数值型答案
    def _is_numeric(ex):
        try:
            float(ex[answer_field])
            return True
        except Exception:
            return False

    train_ds = train_ds.filter(_is_numeric)

    # 调整图像，去掉图像字段，要不打印很多信息
    train_ds = train_ds.map(lambda ex: _resize_rgb(ex, image_field))

    # 构造提示
    def _fmt(ex):
        conv = _vision_prompt_dict(ex[user_field])
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "image": ex["image"], "answer": ex[answer_field]}

    train_ds = train_ds.map(_fmt, remove_columns=train_ds.column_names)

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))

    if eval_ds is not None:
        eval_ds = eval_ds.filter(_is_numeric)
        eval_ds = eval_ds.map(lambda ex: _resize_rgb(ex, image_field))
        eval_ds = eval_ds.map(
            lambda ex: {
                "prompt": tokenizer.apply_chat_template(
                    _vision_prompt_dict(ex[user_field]), tokenize=False, add_generation_prompt=True
                ),
                "image": ex["image"],
                "answer": ex[answer_field],
            },
            remove_columns=eval_ds.column_names,
        )
        if max_eval_samples:
            eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    logger.info("数据准备完成: 训练集=%d, 验证集=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "None")
    return train_ds, eval_ds

# ====== 训练主逻辑 ======

def train(args):
    set_seed(args.seed)
    # 加载模型
    logger.info("加载模型: %s", args.model_name)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
    )

    # 注入 LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=None,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )

    # 数据集
    train_ds, eval_ds = load_and_prepare_dataset(
        dataset_name_or_path=args.dataset,
        tokenizer=tokenizer,
        split_train=args.train_split,
        split_eval=args.eval_split,
        user_field=args.user_field,
        answer_field=args.answer_field,
        image_field=args.image_field,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    # 训练配置
    training_args = GRPOConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        log_completions=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to="none" if not args.report_to else args.report_to,
        output_dir=args.output_dir,
        importance_sampling_level=args.importance_sampling_level,
        mask_truncated_completions=args.mask_truncated_completions,
        loss_type=args.loss_type,
        bf16=args.bf16,
        fp16=args.fp16,
    )
    logger.info(
        "GRPO 配置: logging_steps=%d, save_steps=%d, grad_accu=%d, max_grad_norm=%.3f, scheduler=%s, optim=%s",
        args.logging_steps, args.save_steps, args.gradient_accumulation_steps, args.max_grad_norm,
        args.lr_scheduler_type, args.optim
    )

    # 训练器
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[formatting_reward_func, correctness_reward_func],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("开始训练…")
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("保存模型到: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_lora_dir:
        logger.info("保存 LoRA 适配器到: %s", args.save_lora_dir)
        model.save_lora(args.save_lora_dir)

    if args.push_to_hub:
        repo_id = args.hub_repo_id or os.path.basename(args.output_dir)
        logger.info("推送模型到 HuggingFace Hub: %s", repo_id)
        trainer.push_to_hub(repo_id)

# ====== 命令行参数 ======

def build_arg_parser():
    p = argparse.ArgumentParser(description="仅视觉 GRPO 训练脚本 (适用于 Unsloth Qwen2.5-VL)")

    # 数据相关
    p.add_argument("--dataset", type=str, required=True, help="数据集名称或路径")
    p.add_argument("--train_split", type=str, default="testmini", help="训练集 split 名称")
    p.add_argument("--eval_split", type=str, default=None, help="验证集 split 名称")
    p.add_argument("--user_field", type=str, default="question", help="用户问题字段名")
    p.add_argument("--answer_field", type=str, default="answer", help="答案字段名")
    p.add_argument("--image_field", type=str, default="decoded_image", help="图像字段名")
    p.add_argument("--max_train_samples", type=int, default=None, help="训练集最大样本数（可选）")
    p.add_argument("--max_eval_samples", type=int, default=None, help="验证集最大样本数（可选）")

    # 模型相关
    p.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-VL-3B-Instruct", help="模型名称或路径")
    p.add_argument("--load_in_4bit", type=lambda s: s.lower() in ("1","true","yes"), default=True, help="是否使用 4bit 量化加载模型")
    p.add_argument("--max_seq_length", type=int, default=16384, help="最大序列长度, Qwen2.5这个必须是这个长度16384")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU 显存利用率")
    p.add_argument("--fast_inference", type=lambda s: s.lower() in ("1","true","yes"), default=False, help="是否使用unsloth快速推理")

    # LoRA 设置
    p.add_argument("--lora_r", type=int, default=16, help="LoRA r 值")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha 值")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout 概率")
    p.add_argument("--finetune_vision_layers", type=lambda s: s.lower() in ("1","true","yes"), default=False, help="是否微调视觉层")
    p.add_argument("--finetune_language_layers", type=lambda s: s.lower() in ("1","true","yes"), default=True, help="是否微调语言层")
    p.add_argument("--finetune_attention_modules", type=lambda s: s.lower() in ("1","true","yes"), default=True, help="是否微调注意力模块")
    p.add_argument("--finetune_mlp_modules", type=lambda s: s.lower() in ("1","true","yes"), default=True, help="是否微调 MLP 模块")

    # 训练参数
    p.add_argument("--output_dir", type=str, default="outputs", help="模型输出目录")
    p.add_argument("--save_lora_dir", type=str, default=None, help="LoRA 适配器保存目录")
    p.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减系数")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率 warmup 比例")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    p.add_argument("--optim", type=str, default="adamw_8bit", help="优化器类型")
    p.add_argument("--logging_steps", type=int, default=1, help="日志记录间隔步数")
    p.add_argument("--save_steps", type=int, default=60, help="模型保存间隔步数")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累计步数")
    p.add_argument("--per_device_train_batch_size", type=int, default=1, help="单设备训练批次大小")
    p.add_argument("--num_generations", type=int, default=4, help="每个样本生成次数")
    p.add_argument("--max_prompt_length", type=int, default=1024, help="最大提示长度")
    p.add_argument("--max_completion_length", type=int, default=1024, help="最大生成长度")
    p.add_argument("--num_train_epochs", type=float, default=2, help="训练轮数")
    p.add_argument("--max_steps", type=int, default=100, help="最大训练步数（>0 时覆盖 epochs 设置）")
    p.add_argument("--max_grad_norm", type=float, default=0.1, help="梯度裁剪阈值")
    p.add_argument("--bf16", type=lambda s: s.lower() in ("1","true","yes"), default=False, help="是否启用 bfloat16")
    p.add_argument("--fp16", type=lambda s: s.lower() in ("1","true","yes"), default=False, help="是否启用 float16")
    p.add_argument("--report_to", type=str, default=None, help="日志上报目标（如 wandb、tensorboard）")
    p.add_argument("--importance_sampling_level", type=str, default="sequence", choices=["sequence","token"], help="重要性采样级别")
    p.add_argument("--mask_truncated_completions", type=lambda s: s.lower() in ("1","true","yes"), default=False, help="是否屏蔽截断的生成内容")
    p.add_argument("--loss_type", type=str, default="dr_grpo", help="损失函数类型")

    # 其他
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--push_to_hub", type=lambda s: s.lower() in ("1","true","yes"), default=False, help="是否推送模型到 HuggingFace Hub")
    p.add_argument("--hub_repo_id", type=str, default=None, help="推送到 Hub 的仓库 ID")

    return p


def main():
    """命令行入口：解析参数、创建输出目录并启动训练。"""
    parser = build_arg_parser()
    args = parser.parse_args()
    logger.info("命令行参数解析完成。输出目录: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
