"""Stateless Streamable HTTP MCP endpoint using the official Python SDK."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool

from .catalog import catalog
from .tools import invoke_tool

router = APIRouter(include_in_schema=False)


class McpResponse(Response):
    def __init__(self, request: Request):
        super().__init__()
        self.request = request

    async def __call__(self, scope, receive, send):
        server = Server("llm-d-prism", version="0.1.0")

        @server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name=tool["name"],
                    description=f"[{tool['riskTier']}] {tool['description']}",
                    inputSchema=tool["inputSchema"],
                )
                for tool in catalog(self.request.app)
            ]

        @server.call_tool(validate_input=False)
        async def call_tool(name, arguments):
            try:
                result = await invoke_tool(self.request, name, arguments or {})
                return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])
            except Exception as error:
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({"error": str(error)}))], isError=True
                )

        manager = StreamableHTTPSessionManager(server, stateless=True)
        async with manager.run():
            await manager.handle_request(scope, receive, send)


@router.post("/api/assistant/mcp")
async def mcp(request: Request):
    return McpResponse(request)


@router.api_route("/api/assistant/mcp", methods=["GET", "DELETE"])
async def unsupported_method():
    return JSONResponse(
        {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Method not allowed."}, "id": None}, status_code=405
    )
