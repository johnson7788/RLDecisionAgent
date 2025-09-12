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
import logging

import httpx
from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
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
JSON_CANDIDATE = re.compile(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", re.DOTALL)

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
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_proxy_parse: str = Header("on"),
):
    body = await request.json()
    model = body.get("model")
    stream = bool(body.get("stream", False))

    try:
        route_key, provider = resolve_provider(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    upstream_key = ""
    if authorization and authorization.lower().startswith("bearer "):
        upstream_key = authorization.split(" ", 1)[1].strip()

    if not upstream_key:
        # 明确返回 401，避免客户端误判
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header (Bearer token)")

    headers = {
        "Content-Type": "application/json",
        **provider["auth_header"](upstream_key),
    }
    upstream_body = provider["map_body"](body)

    req_id = f"proxy-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if stream:
            # 使用 stream=True 发送请求，以便先读取状态码和头
            req = client.build_request("POST", provider["base_url"], headers=headers, json=upstream_body)
            r = await client.send(req, stream=True)

            # 如果上游返回非 200，则读取完整错误信息并作为 JSON 返回
            if r.status_code != 200:
                await r.aread()
                try:
                    payload = r.json()
                except Exception:
                    payload = {"error": r.text}
                resp = JSONResponse(status_code=r.status_code, content=payload)
                resp.headers["x-proxy-request-id"] = req_id
                return resp

            # 上游返回 200，正常进行流式转发
            async def event_stream() -> AsyncGenerator[bytes, None]:
                full_text = []
                try:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            payload = line[len("data: "):]
                            if payload == "[DONE]":
                                text_full = "".join(full_text)
                                parsed = extract_json_obj(text_full)
                                PARSED_STORE[req_id] = {
                                    "request_id": req_id,
                                    "parsed": parsed,
                                    "raw_text": text_full,
                                    "ts": time.time(),
                                }
                                yield b"data: [DONE]\n\n"
                                break
                            else:
                                try:
                                    j = json.loads(payload)
                                    choices = j.get("choices") or []
                                    if choices:
                                        delta = choices[0].get("delta") or {}
                                        if delta.get("content") is not None:
                                            full_text.append(delta["content"])
                                except Exception:
                                    pass
                                yield (f"data: {payload}\n\n").encode("utf-8")
                    # 确保即使循环未正常结束（如上游提前关闭），也能保存已收到的文本
                    if req_id not in PARSED_STORE:
                        text_full = "".join(full_text)
                        parsed = extract_json_obj(text_full)
                        PARSED_STORE[req_id] = {
                            "request_id": req_id,
                            "parsed": parsed,
                            "raw_text": text_full,
                            "ts": time.time(),
                        }
                finally:
                    await r.aclose()

            resp = StreamingResponse(event_stream(), media_type="text/event-stream")
            resp.headers["x-proxy-request-id"] = req_id
            resp.headers["cache-control"] = "no-cache"
            return resp

        else: # Non-streaming
            r = await client.post(provider["base_url"], headers=headers, json=upstream_body)
            if r.status_code != 200:
                try:
                    return JSONResponse(status_code=r.status_code, content=r.json())
                except Exception:
                    return JSONResponse(status_code=r.status_code, content={"error": r.text})

            data = r.json()
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
                data["proxy_parsed"] = parsed
            resp = JSONResponse(data)
            resp.headers["x-proxy-request-id"] = req_id
            return resp


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7300)