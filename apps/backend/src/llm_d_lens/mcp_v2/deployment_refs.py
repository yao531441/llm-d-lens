"""Resolve agent-facing deployment ids to Lens execution/run ids and decide ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError


@dataclass(frozen=True)
class DeploymentRef:
    row: Resource | None  # ledger record; None means the deployment was not created through MCP v2
    execution_id: str | None
    run_id: str | None
    foreign_owner: bool = False  # the ledger knows this execution, but under another owner

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
        hint="Call list_resources(kinds=['deployment'], owner='all') to see valid ids.",
    )


def require_owned(ref: DeploymentRef, deployment_id: str, action: str) -> Resource:
    """The ledger row of a deployment the caller created; mutating anything else is refused."""
    if ref.row is None:
        code = "not_owner" if ref.foreign_owner else "owner_unknown"
        reason = "belongs to another user" if ref.foreign_owner else "was not created through this server"
        raise ToolError(
            code,
            f"Deployment '{deployment_id}' {reason}, so it cannot be {action}.",
            field="deployment_id",
            hint="Only deployments you created with create_deployment can be changed; deploy your own copy instead.",
        )
    if ref.row.state == "deleted":
        raise ToolError(
            "conflict_state",
            f"Deployment '{deployment_id}' was deleted.",
            field="deployment_id",
            hint="Create a new deployment, for example create_deployment(from_deployment_id=...).",
        )
    return ref.row


def latest_case(run: dict[str, Any]) -> dict[str, Any] | None:
    """The newest attempt of the run's single case chain (restarts append attempts)."""
    cases = run.get("cases") or []
    return max(cases, key=lambda case: case.get("attempt") or 0) if cases else None


async def get_run(ctx: ToolContext, run_id: str) -> dict[str, Any] | None:
    try:
        return await ctx.api.request("GET", f"/deployments/runs/{run_id}")
    except LensApiError as error:
        if error.status == 404:
            return None
        raise


async def run_execution_id(ctx: ToolContext, run_id: str) -> str | None:
    """The execution id appears on the run's newest case shortly after creation."""
    case = latest_case(await get_run(ctx, run_id) or {})
    return (case or {}).get("execution_id")


async def resolve(ctx: ToolContext, deployment_id: str) -> DeploymentRef:
    """Accepts a ledger id (``dep-...``) or a raw Lens execution id; ownership comes only from the ledger."""
    foreign = False
    if deployment_id.startswith("dep-"):
        row = ctx.ledger.get(deployment_id)
        if row is None or row.owner != ctx.owner or row.kind != "deployment":
            raise not_found(deployment_id)
    else:
        row = ctx.ledger.find_by_execution(deployment_id)
        if row is not None and row.owner != ctx.owner:
            row, foreign = None, True
    if row is None:
        return DeploymentRef(None, deployment_id, None, foreign_owner=foreign)
    execution_id = row.lens_execution_id
    previous = row.meta.get("restarting_from")
    if (execution_id is None or previous) and row.lens_run_id:
        found = await run_execution_id(ctx, row.lens_run_id)
        # After a restart the old execution id stays on the parent case until the retry gets its own.
        execution_id = found if found and found != previous else None
        if execution_id:
            ctx.ledger.bind_execution(row.id, execution_id)
            if previous:
                ctx.ledger.update_meta(row.id, {"restarting_from": None})
    return DeploymentRef(row, execution_id, row.lens_run_id)
