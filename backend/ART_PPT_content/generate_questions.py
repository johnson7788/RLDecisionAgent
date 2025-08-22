#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/22 15:58
# @File  : generate_questions.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 从 train.jsonl 生成“一句话总结+问句”，保存到本地 questions.jsonl

import os
import sys
import json
import time
import argparse
import dotenv
from typing import Dict, Any, Optional
from openai import OpenAI

dotenv.load_dotenv()

def build_client() -> OpenAI:
    """
    使用阿里云 DashScope 的 OpenAI 兼容接口。
    如未配置环境变量，可直接在 OpenAI(...) 中传 api_key="sk-xxx"
    """
    return OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

SYSTEM_PROMPT = (
    "你是一个中文写作助手。请严格遵循以下要求：\n"
    "1) 先用中文将给定内容概括为【一句话】（不超过25个汉字，客观中立，避免口语/人称/夸张；如涉及多位人物，用概称）。\n"
    "2) 再将该一句话概括改写成一个【自然、信息充分的中文问句】，并以中文问号结尾。\n"
    "3) 输出时仅返回严格 JSON，对象形如：{\"summary\":\"…\",\"question\":\"…\"}，不要包含多余文字、标点或解释。\n"
    "4) 若内容噪声较多，也要抓住最主要话题进行提炼；不得输出人身攻击或隐私。"
)

USER_PROMPT_TEMPLATE = (
    "请阅读以下内容，完成“一句话总结”和“问句”两步（严格按系统要求输出 JSON）：\n\n"
    "{content}"
)

def call_llm(client: OpenAI, model: str, content: str, max_retries: int = 3, temperature: float = 0.2) -> Dict[str, str]:
    """
    调用大模型并解析 JSON 输出；带简单重试。
    """
    # 控制上下文长度，避免超长导致报错（按字符简单截断）
    content_trimmed = content.strip()
    if len(content_trimmed) > 4000:
        content_trimmed = content_trimmed[:4000] + "……"

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(content=content_trimmed)}
                ]
            )
            raw = completion.choices[0].message.content.strip()
            # 允许模型外面偶尔包裹反引号等，做一次温和清洗
            raw = raw.strip("` \n\t")
            data = json.loads(raw)
            # 最基本字段校验
            if not isinstance(data, dict):
                raise ValueError("模型输出不是 JSON 对象")
            if "summary" not in data or "question" not in data:
                raise ValueError("模型输出缺少 required 字段：summary/question")
            if not isinstance(data["summary"], str) or not isinstance(data["question"], str):
                raise ValueError("summary/question 必须为字符串")
            return {"summary": data["summary"].strip(), "question": data["question"].strip()}
        except Exception as e:
            last_err = e
            # 指数退避
            sleep_s = min(2 ** attempt, 8)
            time.sleep(sleep_s)
    # 若重试仍失败，抛出最后一次错误
    raise RuntimeError(f"LLM 调用失败：{last_err}")

def process_file(input_path: str, output_path: str, model: str):
    client = build_client()

    total, success, failed = 0, 0, 0

    # 若输出文件已存在，支持在其后追加（也可自行改为覆盖）
    out_f = open(output_path, "a", encoding="utf-8")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    item = json.loads(line)
                    url = item.get("url", "")
                    content = item.get("content", "")
                    if not content:
                        raise ValueError("该行缺少 content 字段或为空")

                    result = call_llm(client, model=model, content=content)

                    out_obj: Dict[str, Any] = {
                        "url": url,
                        "summary": result["summary"],
                        "question": result["question"]
                    }
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    success += 1
                except Exception as e:
                    failed += 1
                    # 将错误也记录到文件，便于排查
                    err_obj = {
                        "error": str(e)[:500],
                        "raw_line": line
                    }
                    out_f.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
    finally:
        out_f.close()

    print(f"处理完成：共 {total} 条，成功 {success} 条，失败 {failed} 条。输出文件：{output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="从 train.jsonl 生成一句话总结与问句")
    parser.add_argument("--in", dest="input_path", default="train_url_content.jsonl", help="输入 JSONL 文件路径")
    parser.add_argument("--out", dest="output_path", default="questions.jsonl", help="输出 JSONL 文件路径")
    parser.add_argument("--model", dest="model", default="qwen-plus", help="模型名称（如 qwen-plus / qwen-turbo 等）")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 环境变量未设置时的友好提示
    if not os.getenv("ALI_API_KEY"):
        print("警告：未检测到 ALI_API_KEY 环境变量，请在 .env 或系统环境中设置，或在 OpenAI(...) 中直接填写 api_key。", file=sys.stderr)
    process_file(args.input_path, args.output_path, model=args.model)
