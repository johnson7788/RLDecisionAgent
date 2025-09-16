#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/16 18:01
# @File  : mcp_client.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : List all MCP tools from an SSE server

import asyncio
import json
from typing import Any, Dict, List
from fastmcp import Client

def tool_definition_to_dict(tool) -> Dict[str, Any]:
    """Converts a  to a dictionary."""
    meta = getattr(tool, "meta", None) or {}
    tags = (meta.get("_fastmcp", {}) or {}).get("tags", [])
    
    tool_dict = {
        "name": tool.name,
        "title": getattr(tool, "title", None),
        "description": getattr(tool, "description", None),
        "inputSchema": getattr(tool, "inputSchema", None),
        "tags": tags
    }
    return {k: v for k, v in tool_dict.items() if v}

async def get_mcp_tools(server_url: str) -> List[Dict[str, Any]]:
    """List all MCP tools from an SSE server and return them as a list of dictionaries."""
    client = Client(server_url)
    async with client:
        try:
            tools = await client.list_tools()
            return [tool_definition_to_dict(t) for t in tools]
        except Exception as e:
            print(f"❌ Failed to get tools from {server_url}: {e}")
            return []

async def main():
    # Example usage: Load config and list tools from all servers
    try:
        with open('../a2a_agent/mcp_config.json', 'r') as f:
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

if __name__ == '__main__':
    asyncio.run(main())