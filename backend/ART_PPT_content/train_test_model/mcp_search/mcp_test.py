#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/26 17:48
# @File  : mcp_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试mcp服务
# python mcp_test.py --transport stdio --server server.py
# pip install mcp click
# mcp_test_fixed.py
# pip install mcp
import asyncio
import argparse
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client  # 仅用于老的 SSE 服务器（已弃用传输）

# 如你的 server 未来改成 Streamable HTTP，可改用：
# from mcp.client.streamable_http import streamablehttp_client  # 见注释

async def run_over_session(read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        print("可用工具：", names)
        assert "search_news" in names, "未发现 search_news 工具"

        print("\n调用 search_news ...\n")
        result = await session.call_tool("search_news", {
            "keyword": "2025年4月 财经 新闻",
            # "count": 5,
            # "search_domain_filter": "www.sohu.com",
            # "search_recency_filter": "noLimit",
            # "content_size": "high",
            # "search_engine": "search_pro",
        })
        if result and result.content:
            for i, item in enumerate(result.content, start=1):
                if getattr(item, "type", "") == "text":
                    print(f"[结果 {i}]\n{item.text}\n" + "-" * 80)
                else:
                    print(f"[结果 {i}] (raw)")
                    print(json.dumps(item, ensure_ascii=False,
                                     default=lambda o: o.__dict__, indent=2))
        else:
            print("没有返回内容。")

async def connect_stdio(server_py: str):
    if not os.path.exists(server_py):
        raise FileNotFoundError(f"找不到 server 脚本：{server_py}")
    params = StdioServerParameters(
        command=sys.executable, args=[server_py, "--transport", "stdio"]
    )
    async with stdio_client(params) as (read, write):
        await run_over_session(read, write)

async def connect_sse(base_url: str):
    base_url = base_url.rstrip("/")
    sse_url = f"{base_url}/sse"
    async with sse_client(sse_url) as (read, write):
        await run_over_session(read, write)

# 若你的服务端改为 Streamable HTTP（例如 http://127.0.0.1:8001/mcp），写法如下：
# async def connect_streamable_http(url: str):
#     async with streamablehttp_client(url) as (read, write, _):
#         await run_over_session(read, write)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--server", help="stdio 模式下的 server.py 路径")
    parser.add_argument("--sse-base", help="SSE 基础地址，默认 http://127.0.0.1:8001")
    args = parser.parse_args()

    if args.transport == "stdio":
        server_py = args.server or os.path.join(
            "server.py"
        )
        await connect_stdio(server_py)
    else:
        await connect_sse(args.sse_base or "http://127.0.0.1:8001")

if __name__ == "__main__":
    asyncio.run(main())
