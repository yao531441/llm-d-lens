"""Streamable HTTP endpoint of the Lens task MCP."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from .context import ToolContext, scopes_from_env
from .lens_api import AsgiLensApi
from .server import build_server

router = APIRouter(include_in_schema=False)


class McpResponse(Response):
    def __init__(self, request: Request):
        super().__init__()
        self.request = request

    async def __call__(self, scope, receive, send):
        async def context() -> ToolContext:
            api = AsgiLensApi(self.request)
            return ToolContext(api=api, owner=await api.principal_id(), scopes=scopes_from_env())

        manager = StreamableHTTPSessionManager(build_server(context), stateless=True)
        async with manager.run():
            await manager.handle_request(scope, receive, send)


@router.post("/api/mcp")
async def mcp(request: Request):
    return McpResponse(request)


@router.api_route("/api/mcp", methods=["GET", "DELETE"])
async def unsupported_method():
    return JSONResponse(
        {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Method not allowed."}, "id": None}, status_code=405
    )
