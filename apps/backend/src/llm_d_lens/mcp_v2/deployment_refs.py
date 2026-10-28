"""Resolve agent-facing deployment ids to Lens execution/run ids and decide ownership."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .context import ToolContext
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError


@dataclass(frozen=True)
class DeploymentRef:
    row: Resource | None  # ledger record; None means the deployment was not created through MCP v2
    execution_id: str | None
    run_id: str | None

    @property
    def owned(self) -> bool:
        return self.row is not None and self.row.state in ("active", "deleting")

    @property
    def public_id(self) -> str:
        return self.row.id if self.row else (self.execution_id or "")


def not_found(deployment_id: str) -> ToolError:
    return ToolError(
        "deployment_not_found",
        f"No deployment '{deployment_id}' is visible to you.",
        field="deployment_id",
        hint="Call list_deployments to see valid ids.",
    )


async def run_execution_id(ctx: ToolContext, run_id: str) -> str | None:
    """The execution id appears on the run's case shortly after creation."""
    try:
        run = await ctx.api.request("GET", f"/deployments/runs/{run_id}")
    except LensApiError as error:
        if error.status == 404:
            return None
        raise
    return next((case["execution_id"] for case in run.get("cases") or [] if case.get("execution_id")), None)


async def resolve(ctx: ToolContext, deployment_id: str) -> DeploymentRef:
    """Accepts a ledger id (``dep-...``) or a raw Lens execution id; ownership comes only from the ledger."""
    if deployment_id.startswith("dep-"):
        row = ctx.ledger.get(deployment_id)
        if row is None or row.owner != ctx.owner or row.kind != "deployment":
            raise not_found(deployment_id)
    else:
        row = ctx.ledger.find_by_execution(deployment_id)
        if row is not None and row.owner != ctx.owner:
            row = None
    if row is None:
        return DeploymentRef(None, deployment_id, None)
    execution_id = row.lens_execution_id
    if execution_id is None and row.lens_run_id:
        execution_id = await run_execution_id(ctx, row.lens_run_id)
        if execution_id:
            ctx.ledger.bind_execution(row.id, execution_id)
    return DeploymentRef(row, execution_id, row.lens_run_id)


async def wait_for_execution(ctx: ToolContext, run_id: str, seconds: float = 20.0) -> str | None:
    deadline = asyncio.get_running_loop().time() + seconds
    while True:
        execution_id = await run_execution_id(ctx, run_id)
        if execution_id or asyncio.get_running_loop().time() >= deadline:
            return execution_id
        await asyncio.sleep(1.0)
