#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/16 18:01
# @File  : mcp_client.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

from fastmcp import Client

# Standard MCP configuration with multiple servers
config = {
    "mcpServers": {
        "energy": {"url": "http://127.0.0.1:9000/sse"},
    }
}

# Create a client that connects to all servers
client = Client(config)

async def main():
    async with client:
        # Access tools and resources with server prefixes
        # forecast = await client.call_tool("energy", {"name": "小明"})
        # print(forecast)
        info = await client.call_tool("get_current_date", {})
        print(info)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())