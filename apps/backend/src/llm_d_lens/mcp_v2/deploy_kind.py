"""The ``deployment`` resource kind: what list_resources and delete_resources do with deployments."""

from __future__ import annotations

from typing import Any

from .context import ToolContext
from .deploy_status import fetch_execution, public_status
from .deployment_refs import get_run, latest_case, resolve, within_create_grace
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError


def _blocked(error: LensApiError) -> ToolError:
    """A Lens 409 on delete: in use by a consumer when Lens says so, otherwise a state conflict.

    delete_resources reports only code and message, so the advice is part of the message.
    """
    if error.code() == "deployment_in_use":
        return ToolError(
            "in_use",
            "An evaluation or another consumer is still using this deployment; wait for it to finish, then delete "
            "again.",
            hint="Wait for it to finish, then delete again.",
        )
    return ToolError(
        "conflict_state",
        f"Lens refused the delete ({error.message()}). Call stop_deployment first and then delete_resources, or "
        "retry in a minute; get_deployment shows the current status.",
        hint="stop_deployment first, then delete_resources; or retry later.",
    )


async def borrowers(ctx: ToolContext, row: Resource) -> list[dict[str, Any]]:
    """Unsettled experiments of this owner that borrow the deployment (they load it with benchmark traffic)."""
    found = []
    for experiment in ctx.ledger.list_resources(owner=ctx.owner, kind="experiment"):
        state = experiment.meta
        if state.get("phase") == "settled":
            continue
        if any(
            variant.get("deploy") == "borrowed" and variant.get("dep_id") == row.id
            for variant in (state.get("variants") or {}).values()
        ):
            found.append(
                {
                    "id": experiment.id,
                    "kind": "experiment",
                    "name": experiment.name,
                    "reason": "borrowed by a running experiment; cancel_experiment it first",
                }
            )
    return found


def borrowed_error(row_id: str, users: list[dict[str, Any]]) -> ToolError:
    ids = ", ".join(user["id"] for user in users)
    return ToolError(
        "in_use",
        f"Deployment {row_id} is used by running experiment(s): {ids}.",
        field="deployment_id",
        hint="Call cancel_experiment first (or wait for it to settle), then retry.",
        dependents=users,
    )


async def delete(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    """Remove the Kubernetes namespace and Lens records. A remote 404 counts as done."""
    if users := await borrowers(ctx, row):
        raise borrowed_error(row.id, users)
    ref = await resolve(ctx, row.id)
    if not ref.run_id and not ref.execution_id:
        current = ctx.ledger.get(row.id) or row
        if current.state == "deleted":
            return
        if within_create_grace(current):
            raise ToolError(
                "in_use",
                "The create request was not confirmed by Lens yet, so there is nothing safe to delete; wait a few "
                "minutes and delete again.",
                hint="Call get_deployment, then retry delete_resources later.",
            )
        return
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
        if error.status == 409:
            raise _blocked(error) from error
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
    """Ledger children plus running experiments that borrow it; Lens reports other consumers only on delete."""
    children = [
        {"id": child.id, "kind": child.kind, "name": child.name, "reason": "created as part of this deployment"}
        for child in ctx.ledger.children(row.id)
    ]
    return [*children, *await borrowers(ctx, row)]


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
