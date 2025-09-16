#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned function-calling model.

Two modes:
1) Unsloth runtime (default): loads a (merged or LoRA-applied) model with FastLanguageModel
2) vLLM runtime (optional): generates raw text with vLLM engine

This script demonstrates:
- providing a tools/functions schema in the system prompt
- asking the model to emit a JSON list of function calls
- parsing the JSON, executing the functions, feeding results back, and
  asking the model for a final natural-language answer.

Environment variables (optional):
- WEATHER_API_KEY (OpenWeatherMap)
- NASA_API_KEY    (api.nasa.gov for APOD via nasapy)
- STOCK_API_KEY   (AlphaVantage)
# 使用
python inference_tool_sft.py --model ./merged_model_16bit \
  --engine unsloth \
  --query "IBM 总部今天的天气如何？顺便把 23 摄氏度转为华氏度。"
# 使用vllm作为推理引擎
python eval_model.py --model your-hf/merged_model_16bit --engine vllm

References:
- Medium article: https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060
"""
import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple

import requests
from datetime import datetime

# Optional dependencies for each engine
try:
    from unsloth import FastLanguageModel
    from transformers import TextStreamer
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False

try:
    import nasapy
    _HAS_NASAPY = True
except Exception:
    _HAS_NASAPY = False


# ----------------------- Demo functions (tools) -----------------------
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY")
STOCK_API_KEY = os.getenv("STOCK_API_KEY")


def get_current_date() -> str:
    """Return current date in YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def get_current_weather(location: str) -> Dict[str, Any]:
    """Get current weather using OpenWeatherMap (metric units)."""
    if not WEATHER_API_KEY:
        return {"error": "WEATHER_API_KEY not set"}
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    try:
        data = requests.get(url, timeout=15).json()
        return {
            "description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
        }
    except Exception as e:
        return {"error": str(e)}


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9.0 / 5.0) + 32.0


def get_nasa_picture_of_the_day(date: str) -> Dict[str, Any]:
    """Fetch NASA APOD for a given date via nasapy (if installed)."""
    if not _HAS_NASAPY:
        return {"error": "nasapy not installed"}
    if not NASA_API_KEY:
        return {"error": "NASA_API_KEY not set"}
    try:
        nasa = nasapy.Nasa(key=NASA_API_KEY)
        apod = nasa.picture_of_the_day(date=date, hd=True)
        return {
            "title": apod.get("title"),
            "explanation": apod.get("explanation"),
            "url": apod.get("url"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_stock_price(ticker: str, date: str) -> Tuple[str, str]:
    """Return (low, high) for ticker on date via AlphaVantage."""
    if not STOCK_API_KEY:
        return ("none", "none")
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={STOCK_API_KEY}"
        data = requests.get(url, timeout=20).json()
        low = data["Time Series (Daily)"][date]["3. low"]
        high = data["Time Series (Daily)"][date]["2. high"]
        return (low, high)
    except Exception:
        return ("none", "none")


AVAILABLE_FUNCTIONS = {
    "get_current_date": get_current_date,
    "get_current_weather": get_current_weather,
    "celsius_to_fahrenheit": celsius_to_fahrenheit,
    "get_nasa_picture_of_the_day": get_nasa_picture_of_the_day,
    "get_stock_price": get_stock_price,
}


def functions_schema() -> List[Dict[str, Any]]:
    """Return JSON schema for tools the model can call."""
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
                        "description": "The city and optional country code, e.g., 'San Francisco, US'",
                    }
                },
                "required": ["location"],
            },
        },
        {
            "name": "celsius_to_fahrenheit",
            "description": "Convert a temperature from Celsius to Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {"celsius": {"type": "number"}},
                "required": ["celsius"],
            },
        },
        {
            "name": "get_nasa_picture_of_the_day",
            "description": "Fetch NASA APOD information for a given date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"],
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


# ----------------------- Inference helpers -----------------------
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    "You are a helpful assistant with access to the following function calls. "
    "Your task is to produce a sequence of function calls necessary to generate "
    "a response to the user utterance. Use the following function calls as required.\n"
    "{available_tools}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)


def extract_assistant_content(text: str) -> str:
    """Extract assistant block between header and <|eot_id|> and strip."""
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def parse_function_calls(text_block: str):
    """Parse JSON list of function calls from the assistant block.
    Be permissive: pick first [ or { and try json.loads.
    """
    idx = min([i for i in [text_block.find("["), text_block.find("{")] if i != -1] or [-1])
    if idx == -1:
        return None
    candidate = text_block[idx:]
    # Truncate on first closing token heuristically
    # but best effort: try loads directly; if fails, progressively trim.
    for end in range(len(candidate), max(idx + 2, len(candidate) - 1), -1):
        try:
            return json.loads(candidate[:end])
        except Exception:
            continue
    try:
        return json.loads(candidate)
    except Exception:
        return None


def run_unsloth(model_name_or_path: str, query: str, max_new_tokens: int = 1024):
    if not _HAS_UNSLOTH:
        raise RuntimeError("Unsloth is not installed. Please install unsloth to use this mode.")

    # Load model for inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path, max_seq_length=2048, dtype=None, load_in_4bit=False
    )
    FastLanguageModel.for_inference(model)

    # Build prompt
    tools = {"functions_str": [json.dumps(x) for x in functions_schema()]}
    input_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": f"You have these tools:\n{tools}"},
         {"role": "user", "content": query}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # First pass: ask for function calls
    out = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, use_cache=True)
    text = tokenizer.batch_decode(out)[0]
    assistant_block = extract_assistant_content(text)
    parsed = parse_function_calls(assistant_block)
    print("=== Raw function-call block ===")
    print(assistant_block)
    print("=== Parsed function calls ===")
    print(parsed)

    # If nothing to call, just print model text
    if not parsed:
        print("No function calls detected. Model output:")
        print(assistant_block)
        return

    # Execute calls, append tool outputs, ask for final answer
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]

    for item in (parsed if isinstance(parsed, list) else [parsed]):
        name = item.get("name")
        args = item.get("arguments", {}) or {}
        fn = AVAILABLE_FUNCTIONS.get(name)
        if not fn:
            tool_response = {"error": f"Unknown tool: {name}"}
        else:
            try:
                tool_response = fn(**args) if isinstance(args, dict) else {"error": "arguments must be an object"}
            except TypeError as te:
                tool_response = {"error": f"Bad arguments for {name}: {te}"}
            except Exception as e:
                tool_response = {"error": str(e)}
        print(f"[Tool:{name}] -> {tool_response}")
        chat.append({
            "role": "tool",
            "content": f"Tool '{name}' with args {args} returned: {tool_response}"
        })

    # Update system prompt to instruct answer synthesis
    chat[0]["content"] = (
        "You are a helpful assistant. Answer the user query based on the provided tool outputs. "
        "Be precise and include relevant numbers/units."
    )

    # Re-run generation with tool outputs in context
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    input_ids2 = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    _ = model.generate(input_ids=input_ids2, streamer=streamer, max_new_tokens=512, use_cache=True,
                       pad_token_id=tokenizer.eos_token_id)


def run_vllm(model_name_or_repo: str, query: str, max_tokens: int = 768):
    if not _HAS_VLLM:
        raise RuntimeError("vLLM is not installed. Please install vllm to use this mode.")

    tools = {"functions_str": [json.dumps(x) for x in functions_schema()]}
    prompt = PROMPT_TEMPLATE.format(available_tools=tools, query=query)

    llm = LLM(model=model_name_or_repo, max_model_len=2048, tokenizer_mode="auto",
              tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.95)
    sp = SamplingParams(max_tokens=max_tokens)
    outputs = llm.generate([prompt], sp)
    text = outputs[0].outputs[0].text
    print("=== vLLM raw output ===")
    print(repr(text))


def main():
    ap = argparse.ArgumentParser(description="Evaluate a function-calling fine-tuned model.")
    ap.add_argument("--model", required=True, help="Path or HF repo (LoRA-merged recommended for vLLM).")
    ap.add_argument("--engine", choices=["unsloth", "vllm"], default="unsloth")
    ap.add_argument("--query", default="What is the current weather in San Francisco, US? Also convert 23°C to Fahrenheit.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    if args.engine == "unsloth":
        run_unsloth(args.model, args.query, args.max_new_tokens)
    else:
        run_vllm(args.model, args.query, min(args.max_new_tokens, 768))


if __name__ == "__main__":
    main()
