#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/12 13:27
# @File  : LLM_json_schema.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import json
import re
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

# === 上游模型路由配置 ===
PROVIDERS = {
    # 例：OpenAI 官方
    "openai:gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "auth_header": lambda key: {"Authorization": f"Bearer {key}"},
        "map_body": lambda body: body,  # 与 OpenAI 完全一致时可直传
        "stream_style": "openai",       # SSE，形如 data: {...}\n\n / data: [DONE]
        "usage_path": ["usage"],        # 非流式时取 token 使用
    },
    # 你也可以加：Azure OpenAI、Claude、Gemini、本地推理服务……
}

# === 简单模型名 -> 提供商路由 ===
MODEL_MAP = {
    "gpt-4o-mini": "openai:gpt-4o-mini",
    # "claude-3-5-sonnet": "anthropic:claude-3-5-sonnet",
    # ...
}

# === 解析策略：从文本里提取 JSON（弱约束示例） ===
JSON_CANDIDATE = re.compile(r"hello", re.DOTALL)

def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从模型回答中提取首个 JSON 对象。
    支持 ```json ...``` 或裸花括号。失败返回 None。
    """
    match = JSON_CANDIDATE.search(text)
    if not match:
        return None
    blob = match.group(1) or match.group(2)
    try:
        return json.loads(blob)
    except Exception:
        # 可在此做宽松修复，如去掉尾逗号、替换单引号等
        try:
            return json.loads(blob.replace("\n", " "))
        except Exception:
            return None

# === 存放解析结果（生产建议用 Redis/DB） ===
PARSED_STORE: Dict[str, Dict[str, Any]] = {}

def resolve_provider(model: str):
    key = MODEL_MAP.get(model)
    if not key:
        raise ValueError(f"Unknown model: {model}")
    return key, PROVIDERS[key]

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str = Header(None),
    x_proxy_parse: str = Header("on"),  # "on"|"off": 非流式时是否在响应里附带解析产物
):
    body = await request.json()
    model = body.get("model")
    stream = bool(body.get("stream", False))

    route_key, provider = resolve_provider(model)

    upstream_key = authorization.split("Bearer ")[-1] if authorization else ""
    headers = {
        "Content-Type": "application/json",
        **provider["auth_header"](upstream_key),
    }
    upstream_body = provider["map_body"](body)

    # 打上链路 ID，便于后续查询解析结果
    req_id = f"proxy-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if stream:
            # === 流式模式：边转发边累积 ===
            async def event_stream() -> AsyncGenerator[bytes, None]:
                full_text = []
                async with client.stream(
                    "POST", provider["base_url"], headers=headers, json=upstream_body
                ) as r:
                    # 透传初始响应头（有限制，只能在开始前设置）
                    # 我们用响应对象外层设置 headers（见 StreamingResponse）
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        # OpenAI 风格：以 'data: ' 开头
                        if line.startswith("data: "):
                            payload = line[len("data: ") :]
                            if payload == "[DONE]":
                                # 存储解析产物
                                text_full = "".join(full_text)
                                parsed = extract_json_obj(text_full)
                                PARSED_STORE[req_id] = {
                                    "request_id": req_id,
                                    "parsed": parsed,
                                    "raw_text": text_full,
                                    "ts": time.time(),
                                }
                                # 结束事件
                                yield b"data: [DONE]\n\n"
                                break
                            else:
                                try:
                                    j = json.loads(payload)
                                    # 从 delta 里累计文本
                                    choices = j.get("choices") or []
                                    if choices:
                                        delta = choices[0].get("delta") or {}
                                        if "content" in delta and delta["content"] is not None:
                                            full_text.append(delta["content"])
                                except Exception:
                                    # 即使上游有偶发非 JSON 行，也保持透传
                                    pass
                                # 原样透传
                                yield (f"data: {payload}\n\n").encode("utf-8")
                    # 保险：若未遇到 [DONE]，也补一次解析存储
                    if req_id not in PARSED_STORE:
                        text_full = "".join(full_text)
                        parsed = extract_json_obj(text_full)
                        PARSED_STORE[req_id] = {
                            "request_id": req_id,
                            "parsed": parsed,
                            "raw_text": text_full,
                            "ts": time.time(),
                        }

            resp = StreamingResponse(event_stream(), media_type="text/event-stream")
            # 在响应头里塞入 proxy 请求 ID，客户端可据此二次拉取解析结果
            resp.headers["x-proxy-request-id"] = req_id
            resp.headers["cache-control"] = "no-cache"
            return resp

        else:
            # === 非流式模式：拿完整 JSON，做解析后可附带返回 ===
            r = await client.post(provider["base_url"], headers=headers, json=upstream_body)
            r.raise_for_status()
            data = r.json()

            # 聚合文本（OpenAI 兼容）
            text_full = "".join([c.get("message", {}).get("content", "") for c in data.get("choices", [])])
            parsed = extract_json_obj(text_full)
            PARSED_STORE[req_id] = {
                "request_id": req_id,
                "parsed": parsed,
                "raw_text": text_full,
                "usage": data.get("usage"),
                "ts": time.time(),
            }

            if (x_proxy_parse or "").lower() == "on":
                data["proxy_parsed"] = parsed  # 注意：部分严格客户端可能不接受未知字段
            resp = JSONResponse(data)
            resp.headers["x-proxy-request-id"] = req_id
            return resp

@app.get("/v1/proxy/parsed/{request_id}")
async def get_parsed(request_id: str):
    item = PARSED_STORE.get(request_id)
    if not item:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return item

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)