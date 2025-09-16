#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/16 18:01
# @File  : mcp_client.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

from fastmcp import Client

# Configuration for the MCP server
# Since we have only one server, we can pass the URL directly.
client = Client("http://127.0.0.1:9000/sse")

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