#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/23 07:45
# @File  : reward_local_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

"""
Test client for Reward Scoring Service
-------------------------------------
既可作为 **命令行脚本**（直接 `python test_reward_api.py`），也可被 **pytest** 发现并执行的测试文件。
默认会读取环境变量 `REWARD_MODEL_URL`（默认为 http://127.0.0.1:8000/score）。

用法（CLI）：
  export REWARD_MODEL_URL=http://127.0.0.1:8000/score
  python test_reward_api.py               # 单次请求
  python test_reward_api.py --burst 20    # 并发 20 次请求（默认并发 10）
  python test_reward_api.py --burst 100 -c 25  # 100 次，请求并发 25
  python test_reward_api.py --health      # 打印 /health

用法（pytest）：
  # 仅在服务已运行时执行：
  TEST_LIVE=1 pytest -q test_reward_api.py

依赖：
  pip install -U httpx pytest anyio
"""
from __future__ import annotations
import dotenv
import os
import json
import time
import math
import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import httpx
dotenv.load_dotenv()

DEFAULT_URL = os.getenv("REWARD_MODEL_URL", "http://127.0.0.1:7000/score")
print(f"使用的REWARD_MODEL_URL: {DEFAULT_URL}")


# -------------------- Sample payloads --------------------

def make_sample(i: int = 0) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    samples = [
        {
            "title": "Show HN: Tiny KV store in 200 lines (with benchmarks)",
            "by": "alice",
            "time": now,
            "scraped_body": "I built a tiny key-value store in ~200 LOC with write-ahead logging and CRC checks.",
            "url": "https://github.com/alice/tiny-kv",
        },
        {
            "title": "Parsing ELF faster than readelf: a zero-copy approach",
            "by": "bob",
            "time": now,
            "scraped_body": "We propose a zero-copy parser for ELF structures using memory maps and typed views.",
            "url": "https://example.com/elf-zero-copy",
        },
        {
            "title": "Why does my TCP connection stall? (notes from a kernel trace)",
            "by": "carol",
            "time": now,
            "scraped_body": "An investigation into TCP stalls using ftrace and tcp_probe to reveal head-of-line blocking.",
            "url": "https://blog.example.net/tcp-stall",
        },
    ]
    return samples[i % len(samples)]


# -------------------- Validation helpers --------------------

def validate_response_json(data: Dict[str, Any]) -> None:
    assert isinstance(data, dict), f"Response must be JSON object, got: {type(data)}"
    assert "score" in data, "Missing 'score'"
    assert "model" in data, "Missing 'model'"
    try:
        score = float(data["score"])  # type: ignore
    except Exception as e:
        raise AssertionError(f"'score' must be a number: {e}")
    assert 0.0 <= score <= 10.0, f"score out of range [0,10]: {score}"
    assert isinstance(data["model"], str) and len(data["model"]) > 0, "'model' must be non-empty string"


async def check_health(base_url: str) -> Dict[str, Any]:
    health_url = base_url.replace("/score", "/health")
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(health_url)
        r.raise_for_status()
        return r.json()


# -------------------- CLI runners --------------------

async def run_once(url: str) -> None:
    payload = make_sample()
    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.perf_counter()
        resp = await client.post(url, json=payload)
        dt = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        validate_response_json(data)
    print("\n[OK] /score response")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Latency: {dt:.1f} ms\n")


async def run_burst(url: str, total: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    scores: List[float] = []

    async def _one(i: int) -> Tuple[float, float]:
        payload = make_sample(i)
        async with sem:
            async with httpx.AsyncClient(timeout=30.0) as client:
                t0 = time.perf_counter()
                r = await client.post(url, json=payload)
                r.raise_for_status()
                dt = (time.perf_counter() - t0) * 1000
                data = r.json()
                validate_response_json(data)
                return dt, float(data["score"])  # type: ignore

    tasks = [asyncio.create_task(_one(i)) for i in range(total)]
    for t in asyncio.as_completed(tasks):
        dt, sc = await t
        latencies.append(dt)
        scores.append(sc)

    def p(vs: List[float], q: float) -> float:
        k = max(0, min(len(vs) - 1, int(q * (len(vs) - 1))))
        return sorted(vs)[k]

    print("\n[OK] Burst summary")
    print(f"Requests: {total} | Concurrency: {concurrency}")
    print(
        "Latency ms - min/median/p95/max: "
        f"{min(latencies):.1f}/{p(latencies,0.5):.1f}/{p(latencies,0.95):.1f}/{max(latencies):.1f}"
    )
    avg = sum(scores) / len(scores) if scores else float("nan")
    print(f"Score   - mean: {avg:.3f} | min: {min(scores):.3f} | max: {max(scores):.3f}\n")


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Test client for Reward Scoring Service")
    parser.add_argument("--url", default=DEFAULT_URL, help="/score endpoint URL")
    parser.add_argument("--burst", type=int, default=0, help="If >0, send N requests concurrently")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrency for burst")
    parser.add_argument("--health", action="store_true", help="Call /health and print result")
    args = parser.parse_args()

    if args.health:
        data = asyncio.run(check_health(args.url))
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if args.burst > 0:
        asyncio.run(run_burst(args.url, args.burst, args.concurrency))
    else:
        asyncio.run(run_once(args.url))


# -------------------- Pytest tests (optional) --------------------
# 仅在设置 TEST_LIVE=1 时运行，以避免在服务未启动时失败。

if os.getenv("PYTEST_CURRENT_TEST"):
    import pytest

    def _skip_if_not_live():
        if os.getenv("TEST_LIVE", "0") != "1":
            pytest.skip("Set TEST_LIVE=1 to run live API tests")

    @pytest.mark.anyio
    async def test_health_live():
        _skip_if_not_live()
        data = await check_health(DEFAULT_URL)
        assert isinstance(data, dict)
        assert "backend" in data

    @pytest.mark.anyio
    async def test_score_live_range():
        _skip_if_not_live()
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(DEFAULT_URL, json=make_sample(1))
            assert r.status_code == 200
            data = r.json()
            validate_response_json(data)


if __name__ == "__main__":
    cli_main()
