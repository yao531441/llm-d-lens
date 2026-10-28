"""Turning a stored plan into a running deployment: ledger row first, then Lens, rolled back on failure."""

from __future__ import annotations

import time
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .ledger import Resource

CREATED_VIA = "lens-mcp-v2"
DEFAULT_TTL_HOURS = 8


def expires_at(ttl_hours: int) -> float:
    return time.time() + ttl_hours * 3600


async def launch_plan(
    ctx: ToolContext,
    plan_id: str,
    *,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    root_id: str | None = None,
    meta: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Resource:
    """Start the deployment of ``plan_id`` (one configuration, one run) and return its bound ledger row."""
    plan = ctx.ledger.get_plan(plan_id, ctx.owner)
    if plan is None:
        raise ToolError(
            "plan_expired",
            f"Plan '{plan_id}' does not exist or has expired.",
            field="plan_id",
            hint="Call plan_deployment again to get a fresh plan_id.",
        )
    name = plan["name"]
    taken = [
        row
        for row in ctx.ledger.list_resources(owner=ctx.owner, kind="deployment", cluster_id=plan["cluster_id"])
        if row.name == name
    ]
    if taken:
        raise ToolError(
            "already_exists",
            f"You already have a deployment named '{name}' on this cluster ({taken[0].id}).",
            field="spec.name",
            hint="Choose another name or omit it to have one generated.",
        )
    row = ctx.ledger.add_resource(
        kind="deployment",
        owner=ctx.owner,
        cluster_id=plan["cluster_id"],
        name=name,
        root_id=root_id,
        state="creating",
        meta={"spec": plan["spec"], "expires_at": expires_at(ttl_hours), **(meta or {})},
    )
    try:
        run = await ctx.api.request(
            "POST",
            "/deployments/runs",
            body={
                "configurations": [plan["deployable_configuration"]],
                "provenance": {
                    "cluster_session_id": plan["cluster_session_id"],
                    "cluster_server_id": plan["cluster_id"],
                    "created_via": CREATED_VIA,
                    "ledger_id": row.id,
                    "plan_id": plan_id,
                    **(provenance or {}),
                },
            },
            timeout=120,
        )
    except BaseException:
        ctx.ledger.set_state(row.id, "deleted")
        raise
    ctx.ledger.bind_run(row.id, run["id"])
    ctx.ledger.set_state(row.id, "active")
    ctx.ledger.drop_plan(plan_id)
    return ctx.ledger.get(row.id)  # type: ignore[return-value]
