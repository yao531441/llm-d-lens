"""The ``deployment`` resource kind: what list_resources and delete_resources do with deployments."""

from __future__ import annotations

from typing import Any

from .context import ToolContext
from .deploy_status import fetch_execution, public_status
from .deployment_refs import get_run, latest_case, resolve
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError


async def delete(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    """Remove the Kubernetes namespace and Lens records. A remote 404 counts as done."""
    ref = await resolve(ctx, row.id)
    try:
        if ref.execution_id:
            try:
                await ctx.api.request(
                    "DELETE",
                    f"/deployments/executions/{ref.execution_id}",
                    params={"delete_namespace": True},
                    timeout=300,
                )
                return
            except LensApiError as error:
                if error.status != 404:
                    raise
        if ref.run_id:
            await ctx.api.request("DELETE", f"/deployments/runs/{ref.run_id}", timeout=300)
    except LensApiError as error:
        if error.status == 404:
            return
        if error.status == 409 and error.code() == "deployment_in_use":
            raise ToolError(
                "in_use",
                "An evaluation or another consumer is still using this deployment.",
                hint="Wait for it to finish or stop it, then delete again.",
            ) from error
        raise


async def describe(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    ref = await resolve(ctx, row.id)
    execution = await fetch_execution(ctx, ref.execution_id) if ref.execution_id else None
    if execution is not None:
        return {
            "status": public_status(execution.get("status")),
            "endpoint": execution.get("endpoint"),
            "model": execution.get("model"),
            "guide": execution.get("guide"),
        }
    case = latest_case(await get_run(ctx, ref.run_id) or {}) if ref.run_id else None
    return {"status": public_status((case or {}).get("status"))}


async def dependents(ctx: ToolContext, row: Resource) -> list[dict[str, Any]]:
    """Resources the ledger parents under this deployment. Lens itself reports evaluation use only on delete."""
    return [
        {"id": child.id, "kind": child.kind, "name": child.name, "reason": "created as part of this deployment"}
        for child in ctx.ledger.children(row.id)
    ]


async def list_others(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    """Executions the caller can see but did not create: 'shared' when another user's ledger has them."""
    data = await ctx.api.request(
        "GET", "/deployments/executions", params={"cluster_id": cluster_id, "limit": 500}, timeout=60
    )
    entries = []
    for execution in data.get("items") or []:
        if public_status(execution.get("status")) == "deleted":
            continue
        known = ctx.ledger.find_by_execution(execution["execution_id"])
        entries.append(
            {
                "id": execution["execution_id"],
                "lens_id": execution["execution_id"],
                "kind": "deployment",
                "name": execution.get("display_name") or execution.get("name"),
                "ownership": "shared" if known is not None else "unknown",
                "state": public_status(execution.get("status")),
                "cluster_id": execution.get("cluster_id"),
                "created_at": execution.get("created_at"),
                "model": execution.get("model"),
                "guide": execution.get("guide"),
            }
        )
    return entries
