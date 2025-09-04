#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-GRPO.ipynb
将基于 Unsloth 的 GRPO 训练流程从 Jupyter 笔记本转换为纯 Python 脚本。
主要步骤：
1) 可选：自动安装依赖（根据是否是 T4 显卡选择 vLLM / Triton 版本）
2) 加载 Qwen3-4B-Base，并应用 LoRA（16bit）
3) 设置自定义“思考 + 答案”聊天模板（<start_working_out> / <end_working_out> / <SOLUTION>）
4) 使用少量高质量样本做预微调（SFT），让模型学会输出格式
5) 准备 Open R1 数学数据集与奖励函数（格式匹配 + 数值比对）
6) 配置并运行 GRPO 训练
7) 推理：对比未加载 LoRA 与加载 LoRA 的输出
8) 保存 LoRA；（可选）演示多种保存/量化方式的代码开关

使用方法：
    python grpo_qwen3_4b.py \
        --run_install 1 --run_sft 1 --run_grpo 1 --run_infer 1 \
        --max_steps 100 --max_seq_length 2048 --lora_rank 32
"""

import os
import re
import gc
import sys
import json
import math
import time
import subprocess
import argparse
from typing import List, Tuple

# -----------------------------
# 命令行参数
# -----------------------------
parser = argparse.ArgumentParser(description="基于 Unsloth 的 GRPO 训练脚本（中文注释版）")
parser.add_argument("--run_install", type=int, default=1, help="是否自动安装依赖（1=是，0=否）")
parser.add_argument("--run_sft", type=int, default=1, help="是否运行预微调阶段（SFT）（1=是，0=否）")
parser.add_argument("--run_grpo", type=int, default=1, help="是否运行 GRPO 训练（1=是，0=否）")
parser.add_argument("--run_infer", type=int, default=1, help="是否在训练后进行推理（1=是，0=否）")
parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度（可增大以容纳更长推理轨迹）")
parser.add_argument("--lora_rank", type=int, default=32, help="LoRA Rank，越大越“聪明”但越慢")
parser.add_argument("--max_steps", type=int, default=100, help="GRPO 训练的最大步数（与 num_train_epochs 二选一）")
parser.add_argument("--save_dir", type=str, default="outputs", help="训练输出目录")
parser.add_argument("--seed", type=int, default=3407, help="随机种子")
parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Base", help="基础模型名称")
args = parser.parse_args()

# -----------------------------
# 实用函数：安全执行 pip 安装
# -----------------------------
def pip_install(pkgs: List[str]):
    """用 pip 安装依赖。"""
    print("📦 正在安装依赖：", " ".join(pkgs))
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs
    subprocess.check_call(cmd)

def detect_gpu_is_t4() -> bool:
    """检测是否为 Tesla T4。"""
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        return "Tesla T4" in out
    except Exception:
        return False

# -----------------------------
# 第 0 步：可选安装依赖
# -----------------------------
if args.run_install:
    print("🔧 [安装] 正在检测环境并安装依赖（如已满足可跳过）...")
    is_t4 = detect_gpu_is_t4()
    # 对 T4 机型固定 vLLM / Triton 版本更稳；其他环境不固定
    get_vllm    = "vllm==0.10.1" if is_t4 else "vllm"
    get_triton  = "triton==3.2.0" if is_t4 else "triton"
    # datasets 限制在 3.x，避免未来 4.x 可能的破坏性变更
    deps_main = [
        "unsloth",
        get_vllm,
        "torchvision",
        "bitsandbytes",
        "xformers",
        get_triton,
        'huggingface_hub>=0.34.0',
        'datasets>=3.4.1,<4.0.0',
        'transformers==4.55.4',
        'safetensors',
        'pandas',
        'numpy',
        'accelerate>=0.30.0',
        'trl>=0.10.0'  # 确保包含 GRPO/SFT 所需
    ]
    pip_install(deps_main)
    print("✅ [安装] 依赖安装完成。\n")

# -----------------------------
# 第 1 步：导入库 & 基本配置
# -----------------------------
print("🧠 正在导入训练所需库...")
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from transformers import TextStreamer
print("✅ 库导入完成。")

# 固定随机种子
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# -----------------------------
# 第 2 步：加载模型并构建 LoRA
# -----------------------------
print("🚀 正在加载基础模型：{} ...".format(args.model_name))
max_seq_length = args.max_seq_length
lora_rank      = args.lora_rank

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False,            # LoRA 用 16-bit
    fast_inference = True,           # 启用 vLLM 推理加速
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,    # 显存紧张可下调
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,      # *2 可加速收敛
    use_gradient_checkpointing = "unsloth",
    random_state = args.seed,
)
print("✅ 模型与 LoRA 初始化完成。\n")

# -----------------------------
# 第 3 步：设置自定义聊天模板（GRPO 所需格式）
# -----------------------------
print("🧩 正在设置自定义聊天模板（含“思考/答案”标记）...")
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {reasoning_start} and {reasoning_end}.\n"
    f"Then, provide your solution between {solution_start}{solution_end}"
)

# Jinja 模板：将 system + 历史对话 + 生成提示 串接起来
chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
    "{% else %}"
        "{{ '" + system_prompt + "' + eos_token }}"
        "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
    "{% endif %}"
)
tokenizer.chat_template = chat_template
print("✅ 聊天模板已设置。\n")

# -----------------------------
# 第 4 步：预微调（SFT）以学习输出格式
# -----------------------------
if args.run_sft:
    print("📚 [SFT] 正在加载用于“学格式”的小样本数据集（unsloth/OpenMathReasoning-mini）...")
    df = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()
    df = df[["expected_answer", "problem", "generated_solution"]]

    # 仅保留答案为“数值”的样本
    is_number = pd.to_numeric(pd.Series(df["expected_answer"]), errors="coerce").notnull()
    df = df.iloc[np.where(is_number)[0]]
    print(f"🔎 [SFT] 原始样本数：{len(df)}")

    def format_dataset_row(x):
        expected_answer = x["expected_answer"]
        problem = x["problem"]
        # 去除 <think> 标签
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_prompt = (
            reasoning_start + thoughts + reasoning_end +
            solution_start + str(expected_answer) + solution_end
        )
        return [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]

    df["Messages"] = df.apply(format_dataset_row, axis=1)

    # 截断到 max_seq_length/2，避免过长轨迹影响格式学习
    print("✂️ [SFT] 正在依据长度截断样本（<= max_seq_length/2）...")
    df["N"] = df["Messages"].apply(lambda msgs: len(tokenizer.apply_chat_template(msgs)))
    df = df.loc[df["N"] <= max_seq_length / 2].copy()
    print(f"✅ [SFT] 截断后样本数：{df.shape[0]}")

    # 构建 HF datasets
    df["text"] = tokenizer.apply_chat_template(df["Messages"].values.tolist(), tokenize=False)
    sft_ds = Dataset.from_pandas(df)

    print("🏋️ [SFT] 正在启动 SFT 训练以让模型学会自定义格式...")
    sft_trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = sft_ds,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 2,
            learning_rate = 2e-4,      # 若长训可降至 2e-5
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = args.seed,
            report_to = "none",
        ),
    )
    sft_trainer.train()
    print("✅ [SFT] 预微调完成。\n")

    # 简单验证：是否遵守模板
    try:
        test_text = tokenizer.apply_chat_template(
            sft_ds[0]["Messages"][:2], tokenize=False, add_generation_prompt=True
        )
        _ = model.generate(
            **tokenizer(test_text, return_tensors="pt").to("cuda"),
            temperature=0,
            max_new_tokens=256,
        )
        print("🔎 [SFT] 模型已学会基本格式（已完成一次无温度采样测试）。")
    except Exception as e:
        print(f"⚠️ [SFT] 测试生成失败：{e}")

    # 释放无用引用
    del sft_ds, df
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("⏭️ 已跳过 SFT 预微调阶段。\n")

# -----------------------------
# 第 5 步：数据准备（Open R1 数学数据集）
# -----------------------------
print("🧮 正在加载 Open R1 数学数据集（open-r1/DAPO-Math-17k-Processed, split=train）...")
r1_ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
print(f"✅ 数据集加载完成，样本数：{len(r1_ds)}")

def extract_hash_answer(text: str) -> str:
    # Open R1 的答案无需从“####”后提取，直接返回
    return text

# 映射为 (prompt, answer)
r1_ds = r1_ds.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),
})
print("✅ 数据字段已转换为 prompt/answer。")

# -----------------------------
# 第 6 步：正则与奖励函数
# -----------------------------
print("🧪 正在构建奖励函数（格式匹配 + 数值提取/比对）...")

solution_end_regex = r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"

match_format = re.compile(
    rf"{re.escape(reasoning_end)}.*?"
    rf"{re.escape(solution_start)}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    """格式完全匹配：+3 分"""
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    """格式部分匹配：每命中一个关键标记 +0.5，缺失或超量则 -1."""
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        # <start_working_out> 在模板中会预置，不奖励
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    """答案比对：完全相等 +5；去空格相等 +3.5；数值比值接近 +1.5~2；否则扣分。"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_resps = [
        g.group(1) if (g := match_format.search(r)) is not None else None
        for r in responses
    ]
    scores = []
    for guess, truth in zip(extracted_resps, answer):
        score = 0.0
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == truth:
            score += 5.0
        elif guess.strip() == str(truth).strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(truth)
                if 0.9 <= ratio <= 1.1:
                    score += 2.0
                elif 0.8 <= ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5
            except Exception:
                score -= 4.5
        scores.append(score)
    return scores

match_numbers = re.compile(
    re.escape(solution_start) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    """从 <SOLUTION> 中提取第一个数字，与真值做 float 比较：相等 +3.5，否则 -1.5。"""
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    extracted = [
        g.group(1) if (g := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    # 每隔若干 step 打印一次观测（中文提示）
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "🧪【观测】\n" +
            "题目:\n{}\n\n标准答案:\n{}\n\n模型输出:\n{}\n\n提取数字:\n{}\n".format(
                question, answer[0], responses[0], extracted[0]
            )
        )
    PRINTED_TIMES += 1

    scores = []
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            t = float(str(truth).strip())
            g = float(str(guess).strip().replace(",", ""))
            scores.append(3.5 if g == t else -1.5)
        except Exception:
            scores.append(0.0)
    return scores

# -----------------------------
# 第 7 步：长度统计，避免超长截断
# -----------------------------
print("📏 正在统计提示长度（保留 90% 分位数以内的样本，以减少截断风险）...")
tokenized = r1_ds.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("✅ 计算得到的 90%% 分位提示长度：", maximum_length)

keep_indices = np.where(np.array(tokenized["L"]) <= maximum_length)[0]
r1_ds = r1_ds.select(keep_indices.tolist())
del tokenized
print(f"✅ 过滤后训练样本数：{len(r1_ds)}\n")

# -----------------------------
# 第 8 步：配置并运行 GRPO 训练
# -----------------------------
if args.run_grpo:
    print("🏁 [GRPO] 正在准备训练参数并启动训练 ...")
    max_prompt_length     = maximum_length + 1
    max_completion_length = max_seq_length - max_prompt_length

    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed  = args.seed,
        stop  = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,  # 可增大为 4 让训练更平滑
        num_generations = 4,               # 显存不足可降低
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = args.max_steps,        # 或者使用 num_train_epochs
        save_steps = args.max_steps,
        report_to = "none",
        output_dir = args.save_dir,
    )

    grpo_trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = r1_ds,
    )

    grpo_trainer.train()
    print("✅ [GRPO] 训练完成。\n")
else:
    print("⏭️ 已跳过 GRPO 训练阶段。\n")

# -----------------------------
# 第 9 步：推理对比（未加载 LoRA vs. 加载 LoRA）
# -----------------------------
if args.run_infer:
    print("🧪 正在进行推理对比（未加载 LoRA / 加载 LoR A）...")
    test_question = "What is the sqrt of 101?"

    # 未加载 LoRA（基座 + vLLM）
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_k = 50,
        max_tokens = 256,
    )
    out_plain = model.fast_generate(
        [test_question],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    print("\n🔹 未加载 LoRA 的输出：\n", out_plain)

    # 保存 LoRA，并加载再推理
    save_lora_dir = os.path.join(args.save_dir, "grpo_saved_lora")
    os.makedirs(save_lora_dir, exist_ok=True)
    model.save_lora(save_lora_dir)
    print(f"\n💾 LoRA 已保存到：{save_lora_dir}")

    # 简单校验 LoRA 参数非全零
    from safetensors import safe_open
    with safe_open(os.path.join(save_lora_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            n_zeros = (t == 0).sum().item()
            assert n_zeros != t.numel(), "检测到 LoRA 权重为全零，可能训练不充分或保存失败。"
    print("✅ LoRA 权重检查通过（非全零）。")

    # 加载 LoRA 后推理
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "What is the sqrt of 101?"},
    ]
    text_for_gen = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    out_lora = model.fast_generate(
        text_for_gen,
        sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=512),
        lora_request = model.load_lora(save_lora_dir),
    )[0].outputs[0].text
    print("\n🔸 加载 LoRA 后的输出：\n", out_lora, "\n")
else:
    print("⏭️ 已跳过推理阶段。\n")

# -----------------------------
# 第 10 步：可选的保存/导出示例（按需打开）
# -----------------------------
print("📦 如需进一步导出为 16bit/4bit/GGUF，请参考以下注释代码，自行切换为 True 即可：")
print("""
# 16bit 合并保存（用于 vLLM）
# if True: model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# 4bit 合并保存
# if True: model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# 仅保存 LoRA 适配器（已在上方保存）
# if True:
#     model.save_pretrained("model")
#     tokenizer.save_pretrained("model")

# GGUF / llama.cpp 导出示例
# if True: model.save_pretrained_gguf("model", tokenizer)                 # q8_0
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
# if True: model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# if True:
#     model.push_to_hub_gguf(
#         "hf/your-model",
#         tokenizer,
#         quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
#         token=""
#     )
""")

print("🎉 全部流程已结束。祝训练顺利！")

