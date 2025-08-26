import asyncio
import json
import os
from typing import Any, Dict
from pathlib import Path
import click
import mcp.types as types
from mcp.server.lowlevel import Server
import dotenv
from zai import ZhipuAiClient

# 兼容两种加载方式
if hasattr(dotenv, "load"):
    dotenv.load()
else:
    try:
        dotenv.load_dotenv()
    except Exception:
        pass

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
assert ZHIPU_API_KEY, "必须提供智谱搜索的API的key"
zhipu_client = ZhipuAiClient(api_key=ZHIPU_API_KEY)

@click.command()
@click.option("--port", default=8001, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    print(f"Starting Search server..., port: {port}, protocol: {transport}")
    app = Server("mcp-search")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_news",
                description="Use ZhipuAi web_search to search the web for content containing the keyword.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword or query to search for (passed to search_query).",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of results to return (1-50). Default 10.",
                        },
                        "search_domain_filter": {
                            "type": "string",
                            "description": "Restrict to a specific domain, e.g. 'www.sohu.com'.",
                        },
                        "search_recency_filter": {
                            "type": "string",
                            "description": "Date range filter, e.g. 'noLimit', 'lastDay', 'lastWeek', etc.",
                        },
                        "content_size": {
                            "type": "string",
                            "description": "Summary size: 'low' | 'medium' | 'high'. Default 'medium'.",
                        },
                        "search_engine": {
                            "type": "string",
                            "description": "Engine name. Default 'search_pro'.",
                        },
                    },
                    "required": ["keyword"],
                },
            ),
        ]

    def _format_web_results(resp: Any, max_items: int) -> str:
        """
        兼容性格式化：尽量提取常见字段，提取不到就直接 JSON pretty 打印。
        """
        def safe_json_dumps(data):
            # Safely dump data to JSON, converting objects to their dict representation.
            return json.dumps(data, ensure_ascii=False, indent=2, default=lambda o: getattr(o, '__dict__', str(o)))

        try:
            results = []
            items = None

            # ZhipuAI SDK v2 returns an object with a 'data' attribute list.
            if hasattr(resp, 'data') and isinstance(resp.data, list):
                items = resp.data
            # Handle dict-based or list-based responses for compatibility.
            elif isinstance(resp, dict):
                for k in ("data", "results", "items", "documents"):
                    if k in resp and isinstance(resp[k], list):
                        items = resp[k]
                        break
            elif isinstance(resp, list):
                items = resp

            if not items:
                return safe_json_dumps(resp)

            for idx, it in enumerate(items[: max_items], start=1):
                is_dict = isinstance(it, dict)
                
                # Extract title, url, and snippet, trying multiple keys/attributes.
                title = (is_dict and (it.get("title") or it.get("page_title") or it.get("name"))) or \
                        getattr(it, "title", getattr(it, "page_title", getattr(it, "name", "Untitled")))
                
                url = (is_dict and (it.get("url") or it.get("source_url") or it.get("link"))) or \
                      getattr(it, "url", getattr(it, "source_url", getattr(it, "link", "")))

                snippet = (is_dict and (it.get("content") or it.get("summary") or it.get("snippet"))) or \
                          getattr(it, "content", getattr(it, "summary", getattr(it, "snippet", "")))

                block = f"{idx}. {title}\n{url}\n{snippet}".strip()
                results.append(block)

            return "\n\n".join(results) if results else safe_json_dumps(resp)
        except Exception:
            # Fallback for any other unexpected errors.
            return safe_json_dumps(resp)

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "search_news":
                keyword = arguments["keyword"]

                # 可选参数与默认
                count = int(arguments.get("count") or 10)
                count = max(1, min(count, 50))
                search_domain_filter = arguments.get("search_domain_filter")  # e.g. "www.sohu.com"
                search_recency_filter = arguments.get("search_recency_filter") or "noLimit"
                content_size = arguments.get("content_size") or "medium"
                search_engine = arguments.get("search_engine") or "search_pro"

                # 调用 ZhipuAi 的 web_search
                resp = zhipu_client.web_search.web_search(
                    search_engine=search_engine,
                    search_query=keyword,
                    count=count,
                    search_domain_filter=search_domain_filter,
                    search_recency_filter=search_recency_filter,
                    content_size=content_size,
                )

                formatted = _format_web_results(resp, max_items=count)

                return [
                    types.TextContent(
                        type="text",
                        text=formatted,
                    )
                ]
            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        asyncio.run(arun())

    return 0


if __name__ == "__main__":
    main()
