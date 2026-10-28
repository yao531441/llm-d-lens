"""Streamable HTTP endpoint for MCP v2, mounted next to (not replacing) ``/api/mcp``."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from .context import ToolContext
from .ledger import Ledger, default_ledger_path
from .lens_api import AsgiLensApi
from .server import build_server

router = APIRouter(include_in_schema=False)

_ledger: Ledger | None = None


def get_ledger() -> Ledger:
    global _ledger
    if _ledger is None:
        _ledger = Ledger(default_ledger_path())
    return _ledger


class McpV2Response(Response):
    def __init__(self, request: Request):
        super().__init__()
        self.request = request

    async def __call__(self, scope, receive, send):
        async def context() -> ToolContext:
            api = AsgiLensApi(self.request)
            return ToolContext(api=api, ledger=get_ledger(), owner=await api.principal_id())

        manager = StreamableHTTPSessionManager(build_server(context), stateless=True)
        async with manager.run():
            await manager.handle_request(scope, receive, send)


@router.post("/api/mcp/v2")
async def mcp_v2(request: Request):
    return McpV2Response(request)


@router.api_route("/api/mcp/v2", methods=["GET", "DELETE"])
async def unsupported_method():
    return JSONResponse(
        {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Method not allowed."}, "id": None}, status_code=405
    )
