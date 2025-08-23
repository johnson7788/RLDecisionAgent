#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/23 07:42
# @File  : local_reward.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 本机的代替远程的reward模型

"""
Reward Scoring Service (FastAPI)
--------------------------------
一个可直接替换你现有 `REWARD_MODEL_URL` 的评分服务：
- `POST /score`：接收故事字段，返回 {"score": float}
- 默认提供 **规则打分**（零依赖、开箱即用）
- 可选 **HuggingFace** 情感打分（pipeline），或 **OpenAI**/兼容 API 的 LLM 打分
- 通过环境变量切换后端与线性标定（A、B）

依赖（任选其一/多）：
- 必需：fastapi, uvicorn, pydantic, python-dotenv
- 可选：transformers, torch (当 SCORER_BACKEND=hf_sst2)
- 可选：openai 或 openai 兼容 SDK（当 SCORER_BACKEND=openai）

运行：
  pip install -U fastapi uvicorn[standard] pydantic python-dotenv
  # 如需 HF：pip install -U transformers torch
  # 如需 OpenAI：pip install -U openai
  uvicorn reward_model_server:app --host 0.0.0.0 --port 8000

示例请求：
  curl -X POST http://127.0.0.1:8000/score \
    -H 'Content-Type: application/json' \
    -d '{
          "title": "Show HN: Tiny KV store in 200 lines",
          "by": "alice",
          "time": "2024-05-03T12:00:00Z",
          "scraped_body": "I built a tiny KV store...",
          "url": "https://github.com/alice/tiny-kv"
        }'

环境变量（.env 支持）：
  SCORER_BACKEND=rule | hf_sst2 | openai         # 默认 rule
  SCORE_A=1.0                                     # 线性标定 y = A*x + B
  SCORE_B=0.0
  SCORE_MIN=0.0                                   # 最终裁剪区间 [MIN, MAX]
  SCORE_MAX=10.0
  # openai 相关（仅在 SCORER_BACKEND=openai 时使用）
  OPENAI_API_KEY=sk-...                           # 或者相容服务的 Key
  OPENAI_BASE_URL=https://api.openai.com/v1       # 可替换为相容服务地址
  OPENAI_MODEL=gpt-4o-mini                         # 模型名
"""

from __future__ import annotations

import math
import os
import re
import logging
from datetime import datetime
from typing import Optional, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# -------------------- Config --------------------
load_dotenv()
logger = logging.getLogger("reward-model")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

SCORER_BACKEND: Literal["rule", "hf_sst2", "openai"] = os.getenv("SCORER_BACKEND", "rule")  # type: ignore
SCORE_A = float(os.getenv("SCORE_A", "1.0"))
SCORE_B = float(os.getenv("SCORE_B", "0.0"))
SCORE_MIN = float(os.getenv("SCORE_MIN", "0.0"))
SCORE_MAX = float(os.getenv("SCORE_MAX", "10.0"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 延迟加载对象（避免无用依赖）
_hf_pipeline = None  # type: ignore
_openai_client = None  # type: ignore

# -------------------- Schemas --------------------

class ScoreRequest(BaseModel):
    title: str = Field(..., description="The title of the story")
    by: str = Field(..., description="The submitter of the story")
    time: str = Field(..., description="ISO 8601 datetime string")
    scraped_body: str = Field(..., description="Body content")
    url: Optional[str] = Field(None, description="Story URL")

    @field_validator("time")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        # 允许任何 ISO 格式，只校验能否被解析
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"time must be ISO 8601: {e}")
        return v

class ScoreResponse(BaseModel):
    score: float
    model: str

# -------------------- Utils --------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

DOMAIN_HINTS = {
    # domain -> bonus
    "github.com": 0.6,
    "arxiv.org": 0.4,
    "docs.google.com": 0.2,
    "medium.com": 0.1,
}

ACTION_VERBS = {
    "build", "built", "create", "created", "rewrite", "rewrote", "design", "designed",
    "release", "released", "open-source", "open sourced", "benchmark", "benchmarks",
}

STOPWORDS = {"a", "an", "the", "and", "or", "to", "of", "in", "for", "on"}

_title_token_re = re.compile(r"[\w\-\+/#]+", re.UNICODE)


def heuristic_score(title: str, body: str, url: Optional[str]) -> float:
    """返回 0-10 的启发式分数。
    规则偏向 HN 风格：信息量、动作性、长度适中、具体性。
    """
    score = 5.0  # 基线

    t = title.strip()
    L = len(t)
    # 长度：30-80 最佳
    if 30 <= L <= 80:
        score += 1.2
    elif 15 <= L < 30 or 80 < L <= 120:
        score += 0.3
    else:
        score -= 1.0

    # 标点与结构
    if ":" in t or "—" in t or "-" in t:
        score += 0.4
    if "?" in t:
        score += 0.2
    if t.lower().startswith("show hn"):
        score += 0.8

    # 具体性：数字/括号
    digits = sum(ch.isdigit() for ch in t)
    if digits >= 2:
        score += 0.5
    if "(" in t and ")" in t:
        score += 0.3

    # 动作动词
    tl = t.lower()
    if any(v in tl for v in ACTION_VERBS):
        score += 0.4

    # 词汇多样性
    tokens = [tok.lower() for tok in _title_token_re.findall(t) if tok]
    uniq = len(set(tokens))
    if tokens:
        type_token_ratio = uniq / max(1, len(tokens))
        score += (type_token_ratio - 0.5)  # 适度鼓励多样性

    # body 信息量
    blen = len(body.strip())
    if 120 <= blen <= 2000:
        score += 0.8
    elif 40 <= blen < 120:
        score += 0.2
    else:
        score -= 0.5

    # 域名加权
    if url:
        try:
            from urllib.parse import urlparse
            netloc = urlparse(url).netloc.lower()
            for dom, bonus in DOMAIN_HINTS.items():
                if dom in netloc:
                    score += bonus
                    break
        except Exception:
            pass

    # 过度大写惩罚
    if t.isupper() and L > 6:
        score -= 1.0

    # 收尾：sigmoid 保稳，再映射至 0-10
    score = 10 * (1 / (1 + math.exp(-0.35 * (score - 5))))
    return _clip(score, 0.0, 10.0)


def hf_sst2_score(title: str, body: str) -> float:
    """使用 HuggingFace sentiment pipeline 给出 0-10 的情感强度分数。
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        try:
            from transformers import pipeline
            _hf_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            logger.info("HF pipeline loaded: distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            logger.error(f"Failed to load HF pipeline: {e}")
            return heuristic_score(title, body, None)

    text = f"Title: {title}\n\nBody: {body[:2000]}"
    try:
        out = _hf_pipeline(text, truncation=True)[0]
        # POSITIVE -> 5~10, NEGATIVE -> 0~5，按置信度线性映射
        conf = float(out.get("score", 0.5))
        if out.get("label") == "POSITIVE":
            raw = 5 + 5 * conf
        else:
            raw = 5 - 5 * conf
        return _clip(raw, 0.0, 10.0)
    except Exception as e:
        logger.error(f"HF inference error: {e}")
        return heuristic_score(title, body, None)


async def openai_score(title: str, body: str) -> float:
    """调用 OpenAI/相容接口，请模型输出 0-10 的整数或小数。"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"OpenAI client init failed: {e}")
            return heuristic_score(title, body, None)

    prompt = (
        "You are a strict evaluator for Hacker News titles.\n"
        "Score the *quality of the title* for attractiveness, clarity, and informativeness, given the body.\n"
        "Return ONLY a number between 0 and 10 (decimals allowed).\n\n"
        f"Title: {title}\n\nBody: {body[:2000]}\n\nScore:"
    )
    try:
        comp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You output only a number."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        text = (comp.choices[0].message.content or "").strip()
        # 提取第一个数字
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        val = float(m.group(0)) if m else 5.0
        return _clip(val, 0.0, 10.0)
    except Exception as e:
        logger.error(f"OpenAI inference error: {e}")
        return heuristic_score(title, body, None)


# -------------------- FastAPI App --------------------
app = FastAPI(title="Reward Scoring Service", version="1.0.0")


@app.get("/health")
async def health():
    return {"ok": True, "backend": SCORER_BACKEND}


@app.post("/score", response_model=ScoreResponse)
async def score_endpoint(req: ScoreRequest):
    # 选择后端
    backend = SCORER_BACKEND

    if backend == "hf_sst2":
        raw = hf_sst2_score(req.title, req.scraped_body)
        model_name = "hf_sst2:distilbert-sst2"
    elif backend == "openai":
        raw = await openai_score(req.title, req.scraped_body)
        model_name = f"openai:{OPENAI_MODEL}"
    else:
        raw = heuristic_score(req.title, req.scraped_body, req.url)
        model_name = "rule:heuristic"

    # 线性标定 + 裁剪
    calibrated = _clip(SCORE_A * raw + SCORE_B, SCORE_MIN, SCORE_MAX)

    return ScoreResponse(score=float(calibrated), model=model_name)


# 便于 `python reward_model_server.py` 直接运行
if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting Reward Scoring Service | backend=%s | A=%.3f B=%.3f | range=[%.1f, %.1f]",
        SCORER_BACKEND,
        SCORE_A,
        SCORE_B,
        SCORE_MIN,
        SCORE_MAX,
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7000")), reload=False)
