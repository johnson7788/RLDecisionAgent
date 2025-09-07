#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 Unsloth + TRL 的 GRPO 在 open-r1/DAPO-Math-17k-Processed 数据集上训练推理格式与答案。
本脚本抽取自 Unsloth 官方示例，专注于 DAPO-Math-17k 的训练部分，并补充中文注释与日志。

运行前环境依赖（建议）：
pip install -U "unsloth" "trl" "vllm" "datasets" "transformers==4.55.4" "bitsandbytes" "xformers" "torchvision"

如果在 Colab 上，建议使用 T4/V100/A100 GPU。
"""

import os
import re
import gc
import math
import time
import json
import random
import logging
from dataclasses import dataclass

import torch
import numpy as np
from datasets import load_dataset, Dataset

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# =========================
# 日志设置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger("GRPO-DAPO-Math")

# =========================
# 可调参数
# =========================
MODEL_NAME = "unsloth/Qwen3-4B-Base"   # 你也可以换成其他基座
MAX_SEQ_LEN = 2048                     # 最大序列长度（越大越吃显存）
LORA_RANK = 32                         # LoRA秩：越大效果越好、训练更慢更耗显存
GPU_MEMORY_UTIL = 0.7                  # vLLM 推理内存利用率
SEED = 3407

# 训练步数与生成配置（GRPO）
MAX_STEPS = 100                        # 演示默认 100 步；正式训练可拉满或改为 num_train_epochs
BATCH_SIZE = 1                         # 每卡 batch；用 grad_accum 可模拟更大 batch
GRAD_ACCUM = 1                         # 梯度累积步数
NUM_GENERATIONS = 4                    # 每个 prompt 生成的并行样本数
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# 数据集（Open R1）
HF_DATASET = "open-r1/DAPO-Math-17k-Processed"
HF_CONFIG = "en"
HF_SPLIT = "train"

# 可选：是否进行一个很小的 SFT 预对齐（来自官方示例，用于让模型更快学会目标格式）
ENABLE_PRE_SFT = False

# 保存 LoRA 的目录
OUTPUT_DIR = "outputs"
LORA_SAVE_DIR = os.path.join(OUTPUT_DIR, "grpo_saved_lora")

# =========================
# 设定随机种子
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# =========================
# 自定义思维/答案标签与系统提示
# =========================
reasoning_start = "<start_working_out>"  # 类似 <think>
reasoning_end   = "<end_working_out>"    # 类似 </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {reasoning_start} and {reasoning_end}.\n"
    f"Then, provide your solution between {solution_start}{solution_end}"
)

# =========================
# 构造 Chat Template（重要）
# - 我们的 GRPO 格式需要模型在生成时以 <start_working_out> 开始
# =========================
def build_chat_template(tokenizer):
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
        "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template
    return tokenizer

# =========================
# （可选）极小 SFT 预对齐以加速 GRPO（来自原笔记本）
# =========================
def maybe_pre_sft(model, tokenizer):
    if not ENABLE_PRE_SFT:
        logger.info("跳过预SFT对齐（可将 ENABLE_PRE_SFT=True 开启）")
        return
    logger.info("开始进行一个小规模的SFT以对齐输出格式（可选步骤）...")
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    ds = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()
    ds = ds[["expected_answer", "problem", "generated_solution"]]

    # 尝试把答案转为数字，过滤失效项（与原示例一致）
    is_number = np.array(pd.to_numeric(ds["expected_answer"], errors="coerce").notnull())
    ds = ds.iloc[np.where(is_number)[0]]

    def format_dataset(x):
        # 去除原有 <think> 标签，贴合我们自定义的标签
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_prompt = reasoning_start + thoughts + reasoning_end + \
                       solution_start + x["expected_answer"] + solution_end
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]

    ds["Messages"] = ds.apply(format_dataset, axis=1)
    ds["text"] = tokenizer.apply_chat_template(ds["Messages"].values.tolist(), tokenize=False)
    from datasets import Dataset as HFDataset
    ds_hf = HFDataset.from_pandas(ds)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_hf,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            report_to="none",
        ),
    )
    trainer.train()
    del ds_hf; gc.collect()
    torch.cuda.empty_cache()
    logger.info("预SFT对齐完成。")

# =========================
# 格式正则与奖励函数（与原示例等价，中文注释）
# =========================
def build_reward_funcs(tokenizer):
    # 允许 </SOLUTION> 后有可选的 EOS
    solution_end_regex = r"</SOLUTION>[\s]{0,}(?:" + re.escape(tokenizer.eos_token) + ")?"

    # 匹配：... <end_working_out> <SOLUTION>答案</SOLUTION> [EOS]
    match_format = re.compile(
        rf"{re.escape(reasoning_end)}.*?"
        rf"{re.escape(solution_start)}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    # 1) 完全匹配格式奖励（匹配到完整结构 +3）
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            resp = completion[0]["content"]
            score = 3.0 if match_format.search(resp) is not None else 0.0
            scores.append(score)
        return scores

    # 2) 近似匹配格式奖励（标签个数正确各 +0.5，错误则 -1）
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            resp = completion[0]["content"]
            score = 0.0
            # start_working_out 由模板自动添加，不再奖励
            score += 0.5 if resp.count(reasoning_end)   == 1 else -1.0
            score += 0.5 if resp.count(solution_start)  == 1 else -1.0
            score += 0.5 if resp.count(solution_end)    == 1 else -1.0
            scores.append(score)
        return scores

    # 3) 基于严格提取答案的奖励（和真值对比，相等 +5，strip后相等 +3.5，
    #    否则尝试转浮点按比例给部分分，失败则扣分）
    def check_answer(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            m.group(1) if (m := match_format.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            score = 0.0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except:
                    score -= 4.5
            scores.append(score)
        return scores

    # 4) 数字提取版（容忍“答案里带文字”，只抽取第一个数字进行对比）
    match_numbers = re.compile(
        re.escape(solution_start) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL,
    )

    # 打印频率（便于观察模型输出）
    PRINT_EVERY_STEPS = 5
    printed_times = {"n": 0}

    def check_numbers(prompts, completions, answer, **kwargs):
        # 仅用第一个样本进行可读性日志打印
        question = prompts[0][-1]["content"]
        responses = [c[0]["content"] for c in completions]
        extracted = [
            m.group(1) if (m := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        if printed_times["n"] % PRINT_EVERY_STEPS == 0:
            logger.info("\n" + "*"*20 +
                        f"\n【题目】\n{question}\n【标准答案】\n{answer[0]}"
                        f"\n【模型输出】\n{responses[0]}"
                        f"\n【提取数字】\n{extracted[0]}\n" + "*"*20)
        printed_times["n"] += 1

        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                ta = float(true_answer.strip())
                ga = float(guess.strip().replace(",", ""))
                scores.append(3.5 if ga == ta else -1.5)
            except:
                scores.append(0.0)
        return scores

    return match_format_exactly, match_format_approximately, check_answer, check_numbers

# =========================
# 主流程
# =========================
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------- 1) 加载模型与Tokenizer（LoRA可训练） --------
    logger.info("加载基座模型与Tokenizer：%s", MODEL_NAME)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,                # LoRA 16bit 训练
        fast_inference=False,               # 关闭内嵌 vLLM
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,          # *2 可略微加快训练
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # -------- 2) 设置 Chat Template --------
    tokenizer = build_chat_template(tokenizer)
    logger.info("Chat template 已设定完成。")

    # （可选）做一个很小的 SFT 预对齐（非必须）
    maybe_pre_sft(model, tokenizer)

    # -------- 3) 加载 DAPO-Math-17k 数据集 --------
    logger.info("加载数据集：%s (%s / %s)", HF_DATASET, HF_CONFIG, HF_SPLIT)
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    logger.info("数据集大小：%d", len(ds))
    logger.info("样例 prompt：%s", ds[0]["prompt"][:200].replace("\n", " "))
    logger.info("样例 solution：%s", ds[0]["solution"][:200].replace("\n", " "))

    # -------- 4) 数据字段映射到我们需要的格式 --------
    #    - prompt: 系统提示 + 用户题目
    #    - answer: 直接使用数据集的 "solution"（该数据集无需像 GSM8K 提取 #### 后的答案）
    def extract_hash_answer(text):
        # Open R1 这个处理版数据无需截 ####，保留原答案
        return text

    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })
    logger.info("字段映射完成，示例：%s", json.dumps(ds[0]["prompt"][-1], ensure_ascii=False)[:300])

    # -------- 5) 构建奖励函数 --------
    reward_funcs = build_reward_funcs(tokenizer)
    logger.info("奖励函数已构建：%s", [f.__name__ for f in reward_funcs])

    # -------- 6) 统计长度并过滤超长样本（避免被截断影响训练） --------
    logger.info("开始统计 token 长度并过滤最长的 10%% 样本...")
    tokenized = ds.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    Ls = np.array(tokenized["L"])
    max_prompt_len_q90 = int(np.quantile(Ls, 0.9))
    logger.info("90%% 分位的 prompt token 长度：%d", max_prompt_len_q90)
    kept_indices = np.where(Ls <= max_prompt_len_q90)[0]
    ds = ds.select(kept_indices.tolist())
    logger.info("过滤后数据集大小：%d（移除了 top 10%% 超长样本）", len(ds))

    # -------- 7) 计算 GRPO 的 prompt / completion 长度预算 --------
    max_prompt_length = max_prompt_len_q90 + 1
    max_completion_length = MAX_SEQ_LEN - max_prompt_length
    logger.info("max_prompt_length=%d, max_completion_length=%d",
                max_prompt_length, max_completion_length)

    # -------- 8) vLLM 采样参数（用于生成候选解） --------
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=SEED,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    # -------- 9) 训练参数（GRPO） --------
    training_args = GRPOConfig(
        use_vllm = True,
        vllm_mode="server",
        vllm_server_base_url="http://127.0.0.1:8000",
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=MAX_STEPS,              # 或者使用 num_train_epochs=1 做完整轮次
        save_steps=MAX_STEPS,
        report_to="none",
        output_dir=OUTPUT_DIR,
        # 你也可以打开评估相关参数
        # fp16_full_eval=True,
        # per_device_eval_batch_size=4,
        # eval_accumulation_steps=1,
        # eval_strategy="steps",
        # eval_steps=50,
    )

    # -------- 10) 构建并启动 GRPO 训练 --------
    logger.info("开始 GRPO 训练...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=list(reward_funcs),
        args=training_args,
        train_dataset=ds,
    )
    trainer.train()
    logger.info("GRPO 训练完成。")

    # -------- 11) （可选）保存 LoRA 适配器 --------
    logger.info("保存 LoRA 适配器到：%s", LORA_SAVE_DIR)
    model.save_lora(LORA_SAVE_DIR)
    logger.info("全部完成，用时 %.1f 分钟。", (time.time() - t0) / 60.0)

    # 提醒：可以用 model.load_lora(LORA_SAVE_DIR) 在推理时加载
    logger.info("提示：推理时可调用 model.load_lora('%s') 进行 LoRA 加载。", LORA_SAVE_DIR)


if __name__ == "__main__":
    main()
