import asyncio
import json
from typing import Any, Dict

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
    print(f"Starting Calculator server..., port: {port}, protocol: {transport}")
    app = Server("mcp-calculator")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="add",
                description="Add two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="subtract",
                description="Subtract two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="multiply",
                description="Multiply two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="divide",
                description="Divide two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "Numerator"},
                        "b": {"type": "number", "description": "Denominator"},
                    },
                    "required": ["a", "b"],
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            a = arguments["a"]
            b = arguments["b"]
            result: float

            if name == "add":
                result = a + b
            elif name == "subtract":
                result = a - b
            elif name == "multiply":
                result = a * b
            elif name == "divide":
                if b == 0:
                    raise ValueError("Cannot divide by zero")
                result = a / b
            else:
                raise ValueError(f"Unknown tool: {name}")

            return [
                types.TextContent(
                    type="text",
                    text=str(result),
                )
            ]

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
