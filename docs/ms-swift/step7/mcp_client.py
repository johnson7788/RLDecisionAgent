#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/16 18:01
# @File  : mcp_client.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : List all MCP tools from an SSE server

import json
import threading
from typing import Any, Dict, List
from fastmcp import Client
import asyncio, nest_asyncio
nest_asyncio.apply()

def clean_none(obj):
    if isinstance(obj, dict):
        return {k: clean_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_none(v) for v in obj]
    elif obj is None:
        return ""  # 保留 Python None，交给 json.dumps 转 ""
    return obj

def remove_title_fields(obj: Any) -> Any:
    """递归删除 dict 或 list 中的 title 字段"""
    if isinstance(obj, dict):
        # 先删除自身的 title
        obj = {k: v for k, v in obj.items() if k != "title"}
        # 递归处理子元素
        return {k: remove_title_fields(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_title_fields(item) for item in obj]
    else:
        return obj

def tool_definition_to_dict(tool) -> Dict[str, Any]:
    """单个工具转成需要的格式."""
    meta = getattr(tool, "meta", None) or {}
    tags = (meta.get("_fastmcp", {}) or {}).get("tags", [])
    parameters = getattr(tool, "inputSchema", None)
    if "title" in parameters:
        # 不需要title字段
        parameters.pop("title")
    if parameters:
        parameters = remove_title_fields(parameters)  # 递归清理
        parameters = clean_none(parameters)
    tool_dict = {
        "name": tool.name,
        "description": getattr(tool, "description", None),
        "parameters": parameters,
    }
    return {k: v for k, v in tool_dict.items() if v is not None}

async def get_mcp_tools(server_url: str) -> List[Dict[str, Any]]:
    """获取MCPserver的所有工具通过SSE协议"""
    client = Client(server_url)
    async with client:
        try:
            tools = await client.list_tools()
            return [tool_definition_to_dict(t) for t in tools]
        except Exception as e:
            print(f"❌ Failed to get tools from {server_url}: {e}")
            return []

async def call_mcp_tool_async(server_url, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    通过 fastmcp.Client 调用 MCP 工具（异步版本）。
    server_or_config: 可为单个 server url / 路径 / FastMCP 实例 / 或包含 "mcpServers" 的 config dict
    """
    client = Client(server_url)
    async with client:
        # fastmcp.Client.call_tool 返回的对象可直接是结果，也可能带 .data 属性
        try:
            result = await client.call_tool(tool_name, arguments or {})
            # 1. 有些返回是 dataclass / Pydantic 模型，尝试转成 dict
            if hasattr(result, "content"):
                # . 如果是 list of Root()
                if isinstance(result.content, list):
                    result = result.content[0].text
        except Exception as e:
            print(f"❌ Failed to call tool {tool_name} from {server_url}: {e}")
            return str(e)

        return result


def call_mcp_tool_sync(server_url, tool_name: str, arguments: Dict[str, Any]) -> str:
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop and loop.is_running():
        # 已经有 loop，直接 run_until_complete
        return loop.run_until_complete(call_mcp_tool_async(server_url, tool_name, arguments))
    else:
        # 没有 loop，用 asyncio.run
        return asyncio.run(call_mcp_tool_async(server_url, tool_name, arguments))

async def main():
    # Example usage: Load config and list tools from all servers
    try:
        with open('./mcp_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found. Please run from the 'mcpserver' directory.")
        return
    
    mcp_servers = config.get("mcpServers", {})
    all_tools = []

    for server_name, server_info in mcp_servers.items():
        if not server_info.get("disabled"):
            url = server_info.get("url")
            if url:
                print(f"🔍 Fetching tools from '{server_name}' at {url}...")
                server_tools = await get_mcp_tools(url)
                if server_tools:
                    print(f"  ✅ Found {len(server_tools)} tools.")
                    all_tools.extend(server_tools)
                else:
                    print("  ⚠️ No tools found or failed to fetch.")
    
    print(f"\n✅ Total tools found: {len(all_tools)}\n")
    for idx, tool in enumerate(all_tools, 1):
        print(f"[{idx}] {tool.get('name')}")
        if tool.get('title'):
            print(f"    标题: {tool.get('title')}")
        if tool.get('description'):
            print(f"    描述: {tool.get('description')}")
        if tool.get('inputSchema'):
            print("    参数Schema:")
            print(json.dumps(tool.get('inputSchema'), ensure_ascii=False, indent=2))
        if tool.get('tags'):
            print(f"    标签: {', '.join(tool.get('tags'))}")
        print()

    result = await call_mcp_tool_async(url, "get_lng_price",  {'region': '浙江', 'start_date': '2025-04-17'})
    print(result)
    result = call_mcp_tool_sync(url, "get_lng_price",  {'region': '浙江', 'start_date': '2025-04-17'})
    print(result)
    result = await call_mcp_tool_async(url, "get_current_time",  {})
    print(result)
    result = call_mcp_tool_sync(url, "get_current_time",  {})
    print(result)
    result = call_mcp_tool_sync(url, "unknow_tools",  {})
    print(result)
if __name__ == '__main__':
    asyncio.run(main())