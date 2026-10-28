"""Standalone stdio server that drives a remote Lens over HTTP.

    LENS_BASE_URL=https://host:5173 LENS_USERNAME=... LENS_PASSWORD=... python -m llm_d_lens.mcp_v2

Useful for trying the tools against a real cluster before the in-process route is deployed.
"""

from __future__ import annotations

import asyncio
import os

from mcp.server.stdio import stdio_server

from .context import ToolContext
from .ledger import Ledger, default_ledger_path
from .lens_api import HttpLensApi
from .server import build_server


async def main() -> None:
    api = HttpLensApi(os.environ["LENS_BASE_URL"], os.environ["LENS_USERNAME"], os.environ["LENS_PASSWORD"])
    ledger = Ledger(default_ledger_path())

    async def context() -> ToolContext:
        return ToolContext(api=api, ledger=ledger, owner=await api.principal_id())

    server = build_server(context)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
