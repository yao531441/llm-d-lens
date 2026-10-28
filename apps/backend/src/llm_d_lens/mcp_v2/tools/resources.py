"""Generic ``list_resources`` and ``delete_resources`` over everything the caller created."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .. import evaluation_ops
from ..context import ToolContext
from ..errors import ToolError
from ..kinds import all_kinds, handler_for
from ..ledger import KIND_PREFIXES, Resource
from ..lens_api import LensApiError
from ..path_ids import PathId
from ..registry import tool

Kind = Literal["deployment", "experiment", "model_cache", "storage_volume", "model_service", "simulation"]


class ListResourcesInput(BaseModel):
    kinds: list[Kind] | None = Field(None, description="Only these kinds; default is every kind.")
    owner: Literal["me", "all"] = Field(
        "me",
        description="'me' lists what you created. 'all' also lists resources you can see but did not create "
        "(read-only; ownership is 'shared' or 'unknown').",
    )
    cluster_id: str | None = Field(None, description="Only resources on this cluster.")
    include: list[Literal["dependents"]] = Field(
        default_factory=list,
        description="'dependents' adds what blocks deletion of each resource, replacing a delete dry-run.",
    )
    include_deleted: bool = Field(False, description="Also show resources already deleted (for audit).")


def _entry(row: Resource) -> dict[str, Any]:
    entry = {
        "id": row.id,
        "kind": row.kind,
        "name": row.name,
        "ownership": "owned",
        "state": row.state,
        "cluster_id": row.cluster_id,
        "created_at": row.created_at,
    }
    if row.meta.get("expires_at"):
        entry["expires_at"] = row.meta["expires_at"]
    if row.meta.get("protected"):
        entry["protected"] = True
    if row.root_id != row.id:
        entry["parent_id"] = row.root_id
    return entry


@tool(
    "list_resources",
    "[R] List the deployments, experiments, caches, volumes, services and simulations you created, with a live "
    "status. owner='all' adds resources created by others or by tools outside this server; those are read-only. "
    "include=['dependents'] shows what blocks deleting each one, so no dry run is needed before delete_resources.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    ListResourcesInput,
    read_only=True,
)
async def list_resources(ctx: ToolContext, args: ListResourcesInput) -> dict[str, Any]:
    kinds = args.kinds or all_kinds()
    entries: list[dict[str, Any]] = []
    for kind in kinds:
        handler = handler_for(kind)
        for row in ctx.ledger.list_resources(
            owner=ctx.owner, kind=kind, cluster_id=args.cluster_id, include_deleted=args.include_deleted
        ):
            entry = _entry(row)
            if handler and handler.describe and row.state != "deleted":
                try:
                    entry.update(await handler.describe(ctx, row))
                except (LensApiError, ToolError) as error:
                    entry["live_status_error"] = getattr(error, "message", None) or str(error)
            if "dependents" in args.include and handler and handler.dependents and row.state != "deleted":
                entry["dependents"] = await handler.dependents(ctx, row)
            entries.append(entry)
        if args.owner == "all" and handler and handler.list_others:
            owned = {row.lens_execution_id for row in ctx.ledger.list_resources(owner=ctx.owner, kind=kind)}
            entries.extend(
                item for item in await handler.list_others(ctx, args.cluster_id) if item.get("lens_id") not in owned
            )
    if "experiment" in kinds:
        try:
            entries.extend(await evaluation_ops.list_experiments(ctx, args.cluster_id, args.owner))
        except (LensApiError, TimeoutError) as error:
            entries.append({"kind": "experiment", "live_status_error": getattr(error, "message", None) or str(error)})
    return {"resources": entries, "count": len(entries)}


class ResourceRef(BaseModel):
    kind: Literal["experiment"] = Field(description="Resources that live only in Lens, such as evaluation workflows.")
    id: PathId


class DeleteResourcesInput(BaseModel):
    ids: list[str] = Field(
        default_factory=list, max_length=50, description="Explicit resource ids, for example dep-1a2b3c4d."
    )
    items: list[ResourceRef] = Field(
        default_factory=list,
        max_length=20,
        description="Lens experiments (evaluation workflows) as {kind: 'experiment', id}. A finished experiment "
        "is deleted with its deployment runs, benchmark history and saved configurations.",
    )
    expected_names: list[str] | None = Field(
        None,
        description="Optional names, one per id followed by one per item; nothing is deleted if any name does "
        "not match its id.",
    )
    cascade: bool = Field(
        False,
        description="Also delete owned resources that depend on these. Without it, a resource that is in use "
        "is reported as blocked with its dependents listed.",
    )
    client_request_id: str | None = Field(None, description="Retry key: repeating it returns the first result.")


@tool(
    "delete_resources",
    "[D] Permanently delete resources you created, by explicit id (1-50) or, for Lens experiments, items "
    "[{kind:'experiment', id}] (a terminal experiment is removed with its deployment runs and history). Reports "
    "per id: deleted, blocked (still in use, protected), not_found or failed; one failure does not stop the "
    "others. Resources you did not "
    "create are refused. Use list_resources(include=['dependents']) first to see what would block. To only "
    "free GPUs and keep the record use stop_deployment instead.\n"
    "Side effects: removes Kubernetes objects and Lens records; resources: frees accelerators/disk; "
    "duration: synchronous, seconds to a minute; reversible: no.",
    DeleteResourcesInput,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def delete_resources(ctx: ToolContext, args: DeleteResourcesInput) -> dict[str, Any]:
    if not args.ids and not args.items:
        raise ToolError("invalid_argument", "Pass ids or items.", field="ids", hint="Use ids from list_resources.")
    if args.expected_names is not None and len(args.expected_names) != len(args.ids) + len(args.items):
        raise ToolError(
            "invalid_argument", "expected_names must have one entry per id and per item.", field="expected_names"
        )
    rows: dict[str, Resource | None] = {resource_id: ctx.ledger.get(resource_id) for resource_id in args.ids}
    for index, resource_id in enumerate(args.ids):
        row = rows[resource_id]
        if args.expected_names and row and row.owner == ctx.owner and row.name != args.expected_names[index]:
            raise ToolError(
                "id_mismatch",
                f"{resource_id} is named '{row.name}', not '{args.expected_names[index]}'. Nothing was deleted.",
                field="expected_names",
                hint="Re-run list_resources and pass the ids and names you actually mean.",
            )
    item_names = (args.expected_names or [])[len(args.ids) :]
    items = {ref.id: name for ref, name in zip(args.items, item_names, strict=False)}
    for item_id, expected in items.items():
        await evaluation_ops.expect_name(ctx, item_id, expected)
    results = [await _delete_one(ctx, resource_id, rows[resource_id], args.cascade) for resource_id in args.ids]
    for item_id in dict.fromkeys(ref.id for ref in args.items):
        results.append(await evaluation_ops.delete(ctx, item_id))
    return {
        "results": results,
        "deleted": [item["id"] for item in results if item["status"] == "deleted"],
        "summary": {
            status: sum(1 for item in results if item["status"] == status)
            for status in ("deleted", "blocked", "not_found", "failed")
        },
    }


async def _delete_one(ctx: ToolContext, resource_id: str, row: Resource | None, cascade: bool) -> dict[str, Any]:
    def outcome(status: str, **extra: Any) -> dict[str, Any]:
        return {"id": resource_id, "status": status, **extra}

    if resource_id.startswith("orph"):
        return outcome("failed", error="scope_required", message="Orphan cleanup needs the cluster-admin profile.")
    if row is None or resource_id.split("-", 1)[0] not in KIND_PREFIXES.values():
        return outcome(
            "not_found",
            error="owner_unknown",
            message="Not a resource created through this server. Resources without a creation record are read-only.",
        )
    if row.owner != ctx.owner:
        return outcome("failed", error="not_owner", message="This resource belongs to another user.")
    if row.state == "deleted":
        return outcome("deleted", already_deleted=True)
    if row.meta.get("protected"):
        return outcome(
            "blocked", error="protected_resource", message="Marked protected; only a human can unprotect it."
        )
    handler = handler_for(row.kind)
    if handler is None:
        return outcome("failed", error="unsupported_kind", message=f"No deleter for {row.kind}.")
    if handler.dependents and not cascade:
        dependents = await handler.dependents(ctx, row)
        if dependents:
            return outcome(
                "blocked",
                error="in_use",
                message="Other resources depend on this one. Delete them in the same call or pass cascade=true.",
                dependents=dependents,
            )
    previous = row.state
    ctx.ledger.set_state(row.id, "deleting")
    try:
        await handler.delete(ctx, row, cascade)
    except ToolError as error:
        ctx.ledger.set_state(row.id, previous)
        return outcome("blocked", error=error.code, message=error.message, **error.extra)
    except LensApiError as error:
        ctx.ledger.set_state(row.id, previous)
        return outcome("failed", error=error.code() or f"lens_{error.status}", message=error.message())
    except Exception as error:  # noqa: BLE001 - leave the row retryable and report which id failed
        ctx.ledger.set_state(row.id, previous)
        return outcome("failed", error="upstream_unavailable", message=f"{type(error).__name__}: {error}"[:300])
    for child in ctx.ledger.children(row.id):
        ctx.ledger.set_state(child.id, "deleted")
    ctx.ledger.set_state(row.id, "deleted")
    return outcome("deleted", kind=row.kind, name=row.name)
