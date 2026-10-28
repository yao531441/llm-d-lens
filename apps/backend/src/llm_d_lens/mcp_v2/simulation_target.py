"""Pick the deployment a simulation may replay a trace against (delta R15: only the caller's own targets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .deployment_refs import run_execution_id
from .errors import ToolError
from .ledger import Resource


@dataclass(frozen=True)
class SimulationTarget:
    row: Resource
    execution_id: str
    cluster_id: str
    endpoint_fields: dict[str, Any]


def _owned_row(ctx: ToolContext, target_id: str) -> Resource:
    if target_id.startswith("svc-"):
        raise ToolError(
            "unsupported_combination",
            "Model-service targets are not supported by start_simulation.",
            field="target_id",
            hint="Lens needs a service access token to replay against a model service and agents never hold one. "
            "Target a deployment you created (dep-...) instead.",
        )
    if target_id.startswith("dep-"):
        row = ctx.ledger.get(target_id)
        if row is None or row.kind != "deployment":
            raise ToolError(
                "not_found", f"No deployment '{target_id}'.", field="target_id", hint="Call list_resources."
            )
    else:
        row = ctx.ledger.find_by_execution(target_id)
        if row is None:
            raise ToolError(
                "owner_unknown",
                f"'{target_id}' was not created through this server, so Lens cannot show that it is yours.",
                field="target_id",
                hint="Only deployments you created can be replayed against. Create your own copy with "
                "create_deployment.",
            )
    if row.owner != ctx.owner:
        raise ToolError(
            "not_owner",
            f"{target_id} belongs to another user; replaying a trace against it would load their service.",
            field="target_id",
            hint="Deploy your own copy with create_deployment and target that.",
        )
    if row.state != "active":
        raise ToolError("conflict_state", f"{target_id} is {row.state}, not active.", field="target_id")
    return row


async def resolve_target(ctx: ToolContext, target_id: str) -> SimulationTarget:
    row = _owned_row(ctx, target_id)
    execution_id = row.lens_execution_id
    if execution_id is None and row.lens_run_id:
        execution_id = await run_execution_id(ctx, row.lens_run_id)
        if execution_id:
            ctx.ledger.bind_execution(row.id, execution_id)
    if not execution_id:
        raise ToolError(
            "conflict_state",
            f"{row.id} has no running deployment yet.",
            field="target_id",
            hint="Poll get_deployment until ready is true.",
        )
    execution = await ctx.api.request("GET", f"/deployments/executions/{execution_id}")
    if execution.get("status") != "ready" or not execution.get("endpoint"):
        raise ToolError(
            "conflict_state",
            f"{row.id} is {execution.get('status')}, not ready.",
            field="target_id",
            hint="Poll get_deployment until ready is true.",
        )
    cluster_id = execution.get("cluster_id") or row.cluster_id or ""
    session_id = execution.get("cluster_session_id")
    if not session_id:
        session = await ctx.api.request("GET", "/cluster/session", params={"clusterId": cluster_id})
        session_id = session["sessionId"]
    return SimulationTarget(
        row=row,
        execution_id=execution_id,
        cluster_id=cluster_id,
        endpoint_fields={
            "endpoint_mode": "deployment",
            "cluster_session_id": session_id,
            "endpoint_namespace": execution.get("namespace"),
            "endpoint_deployment_execution_id": execution_id,
            "endpoint_cluster_id": cluster_id,
            "endpoint_deployment_name": row.name or execution.get("name") or "",
            "endpoint_url": execution["endpoint"],
            "model_name": execution.get("model"),
        },
    )
