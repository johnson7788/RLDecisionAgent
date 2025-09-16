#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/16 18:01
# @File  : mcp_client.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : List all MCP tools from an SSE server

import asyncio
import json
from fastmcp import Client

# 指向你的 FastMCP SSE 服务器（/sse 路径）
client = Client("http://127.0.0.1:9000/sse")

async def main():
    async with client:
        try:
            tools = await client.list_tools()
        except Exception as e:
            print(f"❌ 获取工具失败：{e}")
            return

        if not tools:
            print("（没有可用工具）")
            return

        print(f"✅ 共发现 {len(tools)} 个工具：\n")
        for idx, t in enumerate(tools, 1):
            print(f"[{idx}] {t.name}")
            if getattr(t, "title", None):
                print(f"    标题: {t.title}")
            if getattr(t, "description", None):
                print(f"    描述: {t.description}")

            schema = getattr(t, "inputSchema", None)
            if schema:
                print("    参数Schema:")
                print(json.dumps(schema, ensure_ascii=False, indent=2))

            # FastMCP 约定在 meta._fastmcp.tags 暴露标签（若服务器开启）
            tags = []
            meta = getattr(t, "meta", None) or {}
            tags = (meta.get("_fastmcp", {}) or {}).get("tags", [])
            if tags:
                print(f"    标签: {', '.join(tags)}")

            print()  # 空行分隔

if __name__ == '__main__':
    asyncio.run(main())
