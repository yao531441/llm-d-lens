"""Pick the deployment a simulation may replay a trace against (delta R15: only the caller's own targets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .deployment_refs import require_owned, resolve
from .errors import ToolError


@dataclass(frozen=True)
class SimulationTarget:
    deployment_id: str
    name: str
    execution_id: str
    cluster_id: str
    endpoint_fields: dict[str, Any]


async def resolve_target(ctx: ToolContext, target_id: str) -> SimulationTarget:
    ref = await resolve(ctx, target_id)
    # Replaying a trace loads the target for the whole run, so an agent may only target what it owns.
    require_owned(ref, target_id, "published")
    execution = ref.execution
    if not ref.execution_id or execution.get("status") != "ready" or not execution.get("endpoint"):
        raise ToolError(
            "conflict_state",
            f"{target_id} is {execution.get('status') or 'not started'}, not ready.",
            field="target_id",
            hint="Poll deployment_get until ready is true.",
        )
    cluster_id = execution.get("cluster_id") or ""
    session_id = execution.get("cluster_session_id")
    if not session_id:
        session = await ctx.api.request("GET", "/cluster/session", params={"clusterId": cluster_id})
        session_id = session["sessionId"]
    return SimulationTarget(
        deployment_id=ref.public_id,
        name=execution.get("name") or "",
        execution_id=ref.execution_id,
        cluster_id=cluster_id,
        endpoint_fields={
            "endpoint_mode": "deployment",
            "cluster_session_id": session_id,
            "endpoint_namespace": execution.get("namespace"),
            "endpoint_deployment_execution_id": ref.execution_id,
            "endpoint_cluster_id": cluster_id,
            "endpoint_deployment_name": execution.get("name") or "",
            "endpoint_url": execution["endpoint"],
            "model_name": execution.get("model"),
        },
    )
