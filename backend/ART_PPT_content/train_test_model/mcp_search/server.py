import asyncio
import json
from typing import Any, Dict
from pathlib import Path
import click
import mcp.types as types
from mcp.server.lowlevel import Server


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

    # Load the news data from the jsonl file
    # 计算与当前脚本同目录的 JSONL 路径
    script_dir = Path(__file__).resolve().parent
    jsonl_path = script_dir / "train_url_content.jsonl"
    news_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            news_data.append(json.loads(line))

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_news",
                description="Search for news articles containing the keyword",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Keyword to search for"},
                    },
                    "required": ["keyword"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "search_news":
                keyword = arguments["keyword"]
                for article in news_data:
                    if keyword in article["content"]:
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Found news: {article['url']}\n\n{article['content']}",
                            )
                        ]
                return [types.TextContent(type="text", text="No news found.")]
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