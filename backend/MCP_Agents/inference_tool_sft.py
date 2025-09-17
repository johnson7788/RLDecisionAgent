#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned function-calling model (Qwen3 friendly, LoRA-aware).

Two engines:
1) Unsloth runtime (default): loads a full model or a LoRA adapter on top of a base model
2) vLLM runtime (optional): requires a merged/full model (not a raw LoRA adapter dir)

Features:
- Provide a plain JSON Schema tool list to the model
- Ask the model to emit a JSON array of function calls
- Parse JSON robustly, execute tools, feed results back, and ask for the final answer

Example:
python inference_tool_sft.py \
  --model ./lora_model \
  --base_model unsloth/Qwen3-4B-Instruct-2507 \
  --engine unsloth \
  --query "上海今天的天气如何？" \
  --chat_template qwen-3 \
  --load_in_4bit

If using vLLM, pass a merged model repo/path:
python inference_tool_sft.py --model your-hf/merged_model_16bit --engine vllm \
  --query "北京现在天气如何？把 23 摄氏度转为华氏度"
"""

from __future__ import annotations
import os
import re
import json
import argparse
import hashlib
import random
from typing import Any, Dict, List, Tuple
from datetime import datetime
import torch
from unsloth import FastModel
from transformers import AutoTokenizer, TextStreamer
from peft import PeftModel
from unsloth.chat_templates import get_chat_template as _get_chat_template
from vllm import LLM
from vllm.sampling_params import SamplingParams

# =========================
# Utility & Mocked Tools
# =========================

def _stable_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)

def _rng(seed: str) -> random.Random:
    r = random.Random()
    r.seed(_stable_int(seed))
    return r


def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_current_weather(location: str) -> Dict[str, Any]:
    """Mocked weather. Replace with real API call if desired.
    Returns: {description, temperature(°C), humidity(%)}
    """
    r = _rng(f"weather::{location}")
    descs = [
        "clear sky", "few clouds", "scattered clouds", "broken clouds",
        "shower rain", "light rain", "moderate rain", "thunderstorm",
        "mist", "snow",
    ]
    return {
        "description": r.choice(descs),
        "temperature": round(r.uniform(-10.0, 35.0), 1),
        "humidity": r.randint(20, 95),
    }


def get_stock_price(ticker: str, date: str) -> Tuple[str, str]:
    r = _rng(f"stock::{ticker.upper()}::{date}")
    base = 20 + (sum(ord(c) for c in ticker.upper()) % 80)  # 20~99
    drift = (int(date.replace("-", "")) % 13) - 6         # -6~+6
    price = max(3.0, base + drift + r.uniform(-1.5, 1.5))
    spread = max(0.5, r.uniform(0.5, 5.0))
    low = round(price - spread/2, 2)
    high = round(price + spread/2, 2)
    return (f"{low:.2f}", f"{high:.2f}")


AVAILABLE_FUNCTIONS = {
    "get_current_date": get_current_date,
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
}


def functions_schema() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_current_date",
            "description": "Fetch the current date in the format YYYY-MM-DD.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and optional country code, e.g. 'San Francisco, US'",
                    }
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_stock_price",
            "description": "Retrieve (low, high) stock prices for a ticker on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "date"],
            },
        },
    ]


def functions_schema_json() -> str:
    return json.dumps(functions_schema(), ensure_ascii=False, indent=2)


# =========================
# Robust JSON parsing
# =========================

def parse_function_calls(text: str):
    """Extract a JSON list/object from text.
    - Accepts raw JSON, or fenced code blocks (```json ... ``` / ``` ... ```)
    - Prefers first JSON array; falls back to first JSON object
    Returns a Python object or None.
    """
    if not isinstance(text, str):
        return None

    t = text.strip()
    # Remove surrounding code fences if present
    if t.startswith("```") and t.endswith("```"):
        inner = t.strip("`")
        # remove optional language tag at the start
        inner = re.sub(r"^(json|jsonc|javascript|js|txt)\s+", "", inner, flags=re.IGNORECASE)
        t = inner

    # Try direct JSON first
    try:
        return json.loads(t)
    except Exception:
        pass

    # Find first JSON array
    m = re.search(r"\[(?:.|\s)*\]", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Find first JSON object
    m = re.search(r"\{(?:.|\s)*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


# =========================
# Loading (Unsloth + LoRA)
# =========================

def _file_exists(d: str, name: str) -> bool:
    return os.path.isfile(os.path.join(d, name))


def _is_lora_dir(path: str) -> bool:
    return _file_exists(path, "adapter_config.json") or _file_exists(path, "adapter_model.safetensors")


def _has_full_model(path: str) -> bool:
    if not _file_exists(path, "config.json"):
        return False
    for fname in [
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    ]:
        if _file_exists(path, fname):
            return True
    return False


def _load_tokenizer(model_dir: str, base_model: str | None, chat_template: str | None):
    tok_dir = model_dir if (_file_exists(model_dir, "tokenizer.json") or _file_exists(model_dir, "tokenizer_config.json")) else (base_model or model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    # Optionally force a chat template (e.g., qwen_1_5)
    if chat_template:
        tokenizer = _get_chat_template(tokenizer, chat_template=chat_template)
    return tokenizer


def _ensure_pad_token(tokenizer):
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_model_unsloth(model_path: str, base_model: str | None, max_seq_length: int, load_in_4bit: bool, load_in_8bit: bool):
    if _is_lora_dir(model_path):
        if not base_model:
            raise ValueError("--model points to a LoRA adapter directory; please provide --base_model.")
        base, _ = FastModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            full_finetuning=False,
            token=None,
        )
        model = PeftModel.from_pretrained(base, model_path)
    elif _has_full_model(model_path):
        model, _ = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            full_finetuning=False,
            token=None,
        )
    else:
        raise ValueError(f"Cannot recognize {model_path} as a full model or a LoRA adapter directory.")

    try:
        model.eval()
    except Exception:
        pass

    # Move to CUDA if available
    if torch is not None and torch.cuda.is_available():
        try:
            model.to("cuda")
        except Exception:
            pass
    return model


# =========================
# Inference passes
# =========================

def _apply_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = True):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt" if tokenize else None,
    )


def first_pass_generate_unsloth(model, tokenizer, query: str, max_new_tokens: int) -> str:
    tools_json = functions_schema_json()
    sys_prompt = (
        "你可以调用下列工具（以 JSON Schema 描述）。\n"
        "请先输出一段仅包含 JSON 的内容，格式为一个列表，每个元素是：\n"
        '{"name": "<函数名>", "arguments": {<参数>}}\n'
        "不要输出解释文字或额外内容。\n"
        "工具列表如下：\n" + tools_json
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if hasattr(model, "device"):
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            pass

    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    return decoded[len(prompt_text):]


def second_pass_generate_unsloth(model, tokenizer, query: str, tool_results: List[Dict[str, Any]], max_new_tokens: int):
    synthesis_system = (
        "你是一个严谨的助理。请仅基于提供的工具结果，给出中文答案，包含关键数值与单位。"
    )
    synthesis_user = (
        f"用户提问：{query}\n\n"
        f"工具调用与返回：\n{json.dumps(tool_results, ensure_ascii=False, indent=2)}\n\n"
        "请给出最终回答。"
    )
    messages = [
        {"role": "system", "content": synthesis_system},
        {"role": "user", "content": synthesis_user},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if hasattr(model, "device"):
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        except Exception:
            pass

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    _ = model.generate(**inputs, streamer=streamer, **gen_kwargs)


# vLLM versions

def _ensure_tokenizer_for_prompt_only(chat_template: str | None, model_or_repo: str):
    # We only need tokenizer for producing a textual chat-formatted prompt
    tok = AutoTokenizer.from_pretrained(model_or_repo, use_fast=True)
    if chat_template:
        tok = _get_chat_template(tok, chat_template=chat_template)
    return tok


def _vllm_generate_text(llm: LLM, text_prompt: str, max_tokens: int) -> str:
    sp = SamplingParams(max_tokens=max_tokens)
    outputs = llm.generate([text_prompt], sp)
    return outputs[0].outputs[0].text


def first_pass_generate_vllm(model_repo: str, query: str, max_tokens: int, chat_template: str | None) -> str:
    tok = _ensure_tokenizer_for_prompt_only(chat_template, model_repo)

    tools_json = functions_schema_json()
    sys_prompt = (
        "你可以调用下列工具（以 JSON Schema 描述）。\n"
        "请先输出一段仅包含 JSON 的内容，格式为一个列表，每个元素是：\n"
        '{"name": "<函数名>", "arguments": {<参数>}}\n'
        "不要输出解释文字或额外内容。\n"
        "工具列表如下：\n" + tools_json
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query},
    ]
    text_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=model_repo, max_model_len=2048, tokenizer_mode="auto", tensor_parallel_size=1)
    return _vllm_generate_text(llm, text_prompt, max_tokens)


def second_pass_generate_vllm(model_repo: str, query: str, tool_results: List[Dict[str, Any]], max_tokens: int, chat_template: str | None) -> str:
    tok = _ensure_tokenizer_for_prompt_only(chat_template, model_repo)

    synthesis_system = (
        "你是一个严谨的助理。请仅基于提供的工具结果，给出中文答案，包含关键数值与单位。"
    )
    synthesis_user = (
        f"用户提问：{query}\n\n"
        f"工具调用与返回：\n{json.dumps(tool_results, ensure_ascii=False, indent=2)}\n\n"
        "请给出最终回答。"
    )
    messages = [
        {"role": "system", "content": synthesis_system},
        {"role": "user", "content": synthesis_user},
    ]
    text_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=model_repo, max_model_len=2048, tokenizer_mode="auto", tensor_parallel_size=1)
    return _vllm_generate_text(llm, text_prompt, max_tokens)


# =========================
# Orchestration
# =========================

def run_unsloth(args):
    # 使用unsloth进行模型推理
    tokenizer = _load_tokenizer(args.model, args.base_model, args.chat_template)
    _ensure_pad_token(tokenizer)
    model = _load_model_unsloth(
        model_path=args.model,
        base_model=args.base_model,
        max_seq_length=2048,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Pass 1: ask for function calls
    first = first_pass_generate_unsloth(model, tokenizer, args.query, args.max_new_tokens)
    print("=== Raw function-call block ===\n", first)
    calls = parse_function_calls(first)
    print("=== Parsed function calls ===\n", calls)

    if not calls:
        print("未检测到函数调用 JSON，直接输出：\n" + first)
        return

    # Execute tools
    items = calls if isinstance(calls, list) else [calls]
    tool_results = []
    for item in items:
        name = (item or {}).get("name")
        args_ = (item or {}).get("arguments") or {}
        fn = AVAILABLE_FUNCTIONS.get(name)
        if not fn:
            res = {"error": f"Unknown tool: {name}"}
        else:
            try:
                res = fn(**args_) if isinstance(args_, dict) else {"error": "arguments must be an object"}
            except TypeError as te:
                res = {"error": f"Bad arguments for {name}: {te}"}
            except Exception as e:
                res = {"error": str(e)}
        print(f"[Tool:{name}] -> {res}")
        tool_results.append({"name": name, "arguments": args_, "result": res})

    # Pass 2: synthesis
    second_pass_generate_unsloth(model, tokenizer, args.query, tool_results, max_new_tokens=min(512, args.max_new_tokens))


def run_vllm(args):
    # 使用VLLM进行模型推理
    if _is_lora_dir(args.model):
        raise ValueError("vLLM 不支持直接加载 LoRA 适配器目录；请先 merge 得到完整权重后再用 --engine vllm")

    # Pass 1
    text1 = first_pass_generate_vllm(args.model, args.query, max_tokens=min(768, args.max_new_tokens), chat_template=args.chat_template)
    print("=== vLLM raw output (function calls) ===\n", repr(text1))
    calls = parse_function_calls(text1)
    print("=== Parsed function calls ===\n", calls)

    if not calls:
        print("未检测到函数调用 JSON，直接输出：\n" + text1)
        return

    # Execute tools
    items = calls if isinstance(calls, list) else [calls]
    tool_results = []
    for item in items:
        name = (item or {}).get("name")
        args_ = (item or {}).get("arguments") or {}
        fn = AVAILABLE_FUNCTIONS.get(name)
        if not fn:
            res = {"error": f"Unknown tool: {name}"}
        else:
            try:
                res = fn(**args_) if isinstance(args_, dict) else {"error": "arguments must be an object"}
            except TypeError as te:
                res = {"error": f"Bad arguments for {name}: {te}"}
            except Exception as e:
                res = {"error": str(e)}
        print(f"[Tool:{name}] -> {res}")
        tool_results.append({"name": name, "arguments": args_, "result": res})

    # Pass 2
    text2 = second_pass_generate_vllm(args.model, args.query, tool_results, max_tokens=min(512, args.max_new_tokens), chat_template=args.chat_template)
    print("=== vLLM final answer ===\n", text2)


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Evaluate a function-calling fine-tuned model (Qwen3/Unsloth/LoRA ready).")
    ap.add_argument("--model", required=True, help="Path to full model or LoRA adapter dir, or HF repo.")
    ap.add_argument("--base_model", default=None, help="Base model when --model is a LoRA adapter dir (e.g., unsloth/Qwen3-4B-Instruct-2507)")
    ap.add_argument("--engine", choices=["unsloth", "vllm"], default="unsloth")
    ap.add_argument("--query", default="What is the current weather in San Francisco, US? Also convert 23°C to Fahrenheit.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--chat_template", default=None, help="Force a chat template name (e.g., qwen_1_5). If omitted, use tokenizer's default.")
    ap.add_argument("--load_in_4bit", action="store_true", default=False)
    ap.add_argument("--load_in_8bit", action="store_true", default=False)
    args = ap.parse_args()

    # Boolean adjust: if both are set, prioritize 4bit
    if args.load_in_4bit and args.load_in_8bit:
        args.load_in_8bit = False

    if args.engine == "unsloth":
        run_unsloth(args)
    else:
        run_vllm(args)

if __name__ == "__main__":
    main()
