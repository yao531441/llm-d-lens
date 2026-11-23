"""Generic ``resource_list`` and ``resource_delete`` over the kinds the domains register."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..context import ToolContext
from ..errors import ToolError
from ..kinds import KINDS, handler_for
from ..lens_api import LensApiError
from ..path_ids import PathId
from ..registry import tool

Kind = Literal["deployment", "experiment", "model_cache", "storage_volume", "model_service", "simulation"]


class ListResourcesInput(BaseModel):
    kinds: list[Kind] | None = Field(None, description="Only these kinds; default is every kind.")
    owner: Literal["me", "all"] = Field(
        "me",
        description="'me' lists what you created, for the kinds where Lens records an owner (deployment, experiment, "
        "model_service). 'all' lists everything you can see, including kinds without an owner "
        "(model_cache, storage_volume, simulation) and resources of other users, which are read-only.",
    )
    cluster_id: str | None = Field(None, description="Only resources on this cluster.")


@tool(
    "resource_list",
    "[R] List deployments, experiments, model caches, storage volumes, model services and simulations with a live "
    "status. ownership is 'owned', 'shared' (another user's), 'unknown' (Lens records an owner for the kind but "
    "not for this resource) or 'unrecorded' (Lens records no owner for the kind).\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    ListResourcesInput,
    read_only=True,
)
async def list_resources(ctx: ToolContext, args: ListResourcesInput) -> dict[str, Any]:
    if args.cluster_id:
        known = [item["id"] for item in (await ctx.api.request("GET", "/cluster/clusters")).get("items") or []]
        if args.cluster_id not in known:
            raise ToolError(
                "not_found", f"No cluster {args.cluster_id}.", field="cluster_id", hint="Call cluster_list."
            )
    entries: list[dict[str, Any]] = []
    for kind in args.kinds or KINDS:
        try:
            found = await handler_for(kind).list(ctx, args.cluster_id)
        except (LensApiError, TimeoutError) as error:
            message = error.message() if isinstance(error, LensApiError) else str(error)
            entries.append({"kind": kind, "live_status_error": message})
            continue
        entries.extend(entry for entry in found if args.owner == "all" or entry["ownership"] == "owned")
    result: dict[str, Any] = {"resources": entries, "count": len(entries)}
    if not entries:
        scope = "visible to you" if args.owner == "all" else "created by you"
        where = f" on cluster {args.cluster_id}" if args.cluster_id else ""
        result["empty_note"] = f"No {', '.join(args.kinds or KINDS)} resources {scope}{where}." + (
            " Pass owner='all' to include resources created by others." if args.owner == "me" else ""
        )
    return result


class ResourceRef(BaseModel):
    kind: Kind
    id: PathId
    name: str = Field(
        description="The resource's current name, echoed from resource_list; a mismatch refuses the call."
    )


class DeleteResourcesInput(BaseModel):
    items: list[ResourceRef] = Field(min_length=1, max_length=20)


# The rule below is an agent-specific restriction of this server, not a Lens rule: Lens RBAC cannot express "only your
# own", and a person using the Web UI is not held to it. See docs/design/mcp-agent-task-tools-design.md
# ("Ownership and permissions").
_REFUSED = {
    "shared": ("not_owner", "This resource belongs to another user."),
    "unknown": ("owner_unknown", "Lens did not record an owner for this resource."),
}


@tool(
    "resource_delete",
    "[D] Permanently delete explicitly listed resources, each as {kind, id, name} with the name echoed from "
    "resource_list (1-20 items). A name mismatch refuses the whole call before anything is deleted. Only resources "
    "you own are deleted for kinds where Lens records an owner. Reports per item: deleted, blocked (still in use), "
    "not_found or failed; one failure does not stop the others. A finished experiment is deleted with its preserved "
    "deployments and history; storage volumes are deleted without their model files. To only free accelerators "
    "use deployment_stop.\n"
    "Side effects: removes Kubernetes objects and Lens records; resources: frees accelerators/disk; "
    "duration: synchronous, seconds to a minute; reversible: no.",
    DeleteResourcesInput,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def delete_resources(ctx: ToolContext, args: DeleteResourcesInput) -> dict[str, Any]:
    entries = [await handler_for(item.kind).get(ctx, item.id) for item in args.items]
    for item, entry in zip(args.items, entries, strict=True):
        if entry is not None and entry["name"] != item.name:
            raise ToolError(
                "id_mismatch",
                f"{item.kind} {item.id} is named '{entry['name']}', not '{item.name}'. Nothing was deleted.",
                field="items",
                hint="Re-run resource_list and pass the ids and names you actually mean.",
            )
    results = [await _delete_one(ctx, item, entry) for item, entry in zip(args.items, entries, strict=True)]
    return {
        "results": results,
        "summary": {
            status: sum(1 for result in results if result["status"] == status)
            for status in ("deleted", "blocked", "not_found", "failed")
        },
    }


async def _delete_one(ctx: ToolContext, item: ResourceRef, entry: dict[str, Any] | None) -> dict[str, Any]:
    def outcome(status: str, **extra: Any) -> dict[str, Any]:
        return {"kind": item.kind, "id": item.id, "name": item.name, "status": status, **extra}

    if entry is None:
        return outcome("not_found")
    if entry["ownership"] in _REFUSED:
        error, message = _REFUSED[entry["ownership"]]
        return outcome("failed", error=error, message=message)
    try:
        also_deletes = await handler_for(item.kind).delete(ctx, item.id)
    except ToolError as error:
        return outcome("blocked", error=error.code, message=error.message, **error.extra)
    except LensApiError as error:
        return outcome(
            "blocked" if error.status == 409 else "failed",
            error=error.code() or f"lens_{error.status}",
            message=error.message(),
        )
    except TimeoutError:
        return outcome("failed", error="timeout", message="Lens did not answer in time; check the resource and retry.")
    return outcome("deleted", also_deletes=also_deletes)
