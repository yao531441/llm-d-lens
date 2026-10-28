"""Builds the MCP ``Server`` from the tool registry; shared by the HTTP route and the stdio entry point."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from mcp.server.lowlevel import Server
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from .context import ToolContext
from .registry import all_tools, call_tool

ContextFactory = Callable[[], Awaitable[ToolContext]]

SERVER_NAME = "llm-d-lens-v2"


def build_server(context: ContextFactory) -> Server:
    server = Server(SERVER_NAME, version="2.0.0")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.input_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=spec.read_only,
                    destructiveHint=spec.destructive,
                    idempotentHint=spec.idempotent,
                    openWorldHint=spec.open_world,
                ),
            )
            for spec in all_tools()
        ]

    @server.call_tool(validate_input=False)
    async def handle(name, arguments):
        payload, is_error = await call_tool(await context(), name, arguments or {})
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload, indent=2))], isError=is_error)

    return server
