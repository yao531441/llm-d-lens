"""MCP adapter for Lens-owned persisted deployment verifications."""

from __future__ import annotations

import asyncio
from typing import Any

from .context import ToolContext
from .deployment_refs import DeploymentRef
from .errors import ToolError
from .lens_api import LensApiError
from .path_ids import check_id

PREFIX_REQUESTS = 6


async def run_test(
    ctx: ToolContext,
    ref: DeploymentRef,
    checks: list[str],
    requests: int,
    verification_id: str | None = None,
    wait_seconds: int = 25,
) -> dict[str, Any]:
    path = f"/deployments/executions/{ref.execution_id}/verifications"
    if verification_id:
        check_id(verification_id, "verification_id")
        record = await ctx.api.request("GET", f"{path}/{verification_id}")
    else:
        try:
            record = await ctx.api.request("POST", path, body={"checks": checks, "requests": requests})
        except LensApiError as error:
            if error.status == 409:
                raise ToolError("conflict_state", error.message(), field="deployment_id") from error
            raise
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while record["status"] in {"queued", "running"}:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(1, remaining))
        record = await ctx.api.request("GET", f"{path}/{record['id']}")
    return {
        **record,
        "deployment_id": ref.public_id,
        "verification_id": record["id"],
        "next": "Read this task with deployment_test(verification_id=...) or open Web deployment details.",
    }
