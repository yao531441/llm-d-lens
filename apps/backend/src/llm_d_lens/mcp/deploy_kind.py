"""The ``deployment`` resource kind: what resource_list and resource_delete do with deployments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .context import ToolContext
from .deploy_status import public_status
from .deployment_refs import require_owned, resolve
from .errors import ToolError
from .lens_api import LensApiError


def _ownership(execution: dict[str, Any]) -> str:
    return {"self": "owned", "other": "shared"}.get(execution.get("ownership") or "", "unknown")


def _entry(execution: dict[str, Any], deployment_id: str) -> dict[str, Any]:
    return {
        "id": deployment_id,
        "kind": "deployment",
        "name": execution.get("display_name") or execution.get("name"),
        "status": public_status(execution.get("status")),
        "ownership": _ownership(execution),
        "capabilities": execution.get("capabilities") or {},
        "cluster_id": execution.get("cluster_id"),
        "created_at": execution.get("created_at"),
        "run_id": execution.get("run_id"),
        "execution_id": execution.get("execution_id"),
        "model": execution.get("model"),
        "guide": execution.get("guide"),
        "endpoint": execution.get("endpoint"),
    }


async def list_deployments(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    data = await ctx.api.request(
        "GET", "/deployments/executions", params={"cluster_id": cluster_id, "limit": 500}, timeout=60
    )
    items = [item for item in data.get("items") or [] if public_status(item.get("status")) != "deleted"]
    per_run = Counter(item.get("run_id") for item in items)
    # A run id names the deployment when the run has one execution; a run with several needs the execution id.
    return [
        _entry(item, item["run_id"] if item.get("run_id") and per_run[item["run_id"]] == 1 else item["execution_id"])
        for item in items
    ]


async def get_deployment(ctx: ToolContext, deployment_id: str) -> dict[str, Any] | None:
    try:
        ref = await resolve(ctx, deployment_id)
    except ToolError as error:
        if error.code == "deployment_not_found":
            return None
        raise
    return _entry({"execution_id": ref.execution_id, "run_id": ref.run_id, **ref.execution}, ref.public_id)


async def delete_deployment(ctx: ToolContext, deployment_id: str) -> list[dict[str, str]]:
    """Remove the Kubernetes namespace and the Lens execution; the namespace goes with it."""
    ref = await resolve(ctx, deployment_id)
    require_owned(ref, deployment_id, "deleted")
    try:
        await ctx.api.request(
            "DELETE", f"/deployments/executions/{ref.execution_id}", params={"delete_namespace": True}, timeout=300
        )
    except LensApiError as error:
        # A missing execution is already gone; deleting its parent run could take other cases with it.
        if error.status == 404:
            return []
        if error.status == 409:
            raise _blocked(error) from error
        raise
    return []


def _blocked(error: LensApiError) -> ToolError:
    if error.code() == "deployment_in_use":
        return ToolError(
            "in_use",
            "An evaluation or another consumer is still using this deployment; wait for it to finish, then delete "
            "again.",
        )
    return ToolError(
        "conflict_state",
        f"Lens refused the delete ({error.message()}). Call deployment_stop first and then resource_delete, or "
        "retry in a minute; deployment_get shows the current status.",
    )
