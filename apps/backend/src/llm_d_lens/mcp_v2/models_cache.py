"""Model-cache and storage-volume views shared by the model tools, ``get_cluster`` and the ``model_cache`` kind.

Lens has no owner tag on cache entries or storage volumes, so ownership always comes from the local ledger.
"""

from __future__ import annotations

from typing import Any

from .context import ToolContext
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError
from .registry import upstream_error

DEFAULT_REVISION = "main"
# Highest first: the best state of any volume wins when a model is cached on several.
_STATUS_ORDER = ["ready", "downloading", "pending", "failed", "deleting"]
_ACTIVE_DEPLOYMENT = {"deploying", "ready", "rolling_back"}


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("items")
    return [item for item in payload or [] if isinstance(item, dict)]


async def list_volumes(ctx: ToolContext, cluster_id: str) -> list[dict[str, Any]]:
    return _items(await ctx.api.request("GET", "/storage/volumes", params={"cluster_id": cluster_id}, timeout=30))


def is_model_cache_volume(volume: dict[str, Any]) -> bool:
    return "model-cache" in (volume.get("purposes") or [])


def ledger_volume_id(ctx: ToolContext, lens_volume_id: str | None) -> str | None:
    """The caller's ``vol-`` ledger id for a Lens storage volume, if the caller created it."""
    if not lens_volume_id:
        return None
    row = ctx.ledger.find_by_execution(lens_volume_id)
    return row.id if row and row.kind == "storage_volume" and row.owner == ctx.owner else None


def volume_view(ctx: ToolContext, volume: dict[str, Any]) -> dict[str, Any]:
    view = {
        "volume_id": volume.get("id"),
        "name": volume.get("name"),
        "kind": volume.get("kind"),
        "status": volume.get("status"),
        "capacity": volume.get("capacity"),
        "used_bytes": volume.get("usedBytes"),
        "model_caches": volume.get("modelCacheCount"),
        "in_use_by": volume.get("inUseCount"),
        "model_cache": is_model_cache_volume(volume),
    }
    owned = ledger_volume_id(ctx, volume.get("id"))
    if owned:
        view["storage_volume_id"] = owned
    return view


async def resolve_volume(ctx: ToolContext, cluster_id: str, volume_id: str | None) -> dict[str, Any]:
    """Pick the ready model-cache volume to download into, from a Lens id, a ``vol-`` ledger id or the only choice."""
    volumes = await list_volumes(ctx, cluster_id)
    usable = [volume for volume in volumes if is_model_cache_volume(volume) and volume.get("status") == "ready"]
    if volume_id:
        lens_id = volume_id
        if volume_id.startswith("vol-"):
            row = ctx.ledger.get(volume_id)
            if row is None or row.owner != ctx.owner or row.kind != "storage_volume" or not row.lens_execution_id:
                raise ToolError("not_found", f"No volume {volume_id} that you created.", field="volume_id")
            lens_id = row.lens_execution_id
        for volume in volumes:
            if volume.get("id") == lens_id:
                if not is_model_cache_volume(volume):
                    raise ToolError(
                        "invalid_argument",
                        f"Volume {volume_id} is not a model-cache volume.",
                        field="volume_id",
                        hint="Pick a volume with the model-cache purpose; get_cluster(include=['storage']) lists them.",
                    )
                if volume.get("status") != "ready":
                    raise ToolError(
                        "prerequisite_missing",
                        f"Volume {volume_id} is {volume.get('status')}, not ready.",
                        field="volume_id",
                    )
                return volume
        raise ToolError("not_found", f"No volume {volume_id} on cluster {cluster_id}.", field="volume_id")
    if len(usable) == 1:
        return usable[0]
    if not usable:
        raise ToolError(
            "prerequisite_missing",
            f"Cluster {cluster_id} has no ready model-cache volume.",
            field="volume_id",
            hint="Create a storage volume with the model-cache purpose first, then retry.",
        )
    raise ToolError(
        "invalid_argument",
        f"Cluster {cluster_id} has {len(usable)} ready model-cache volumes; choose one.",
        field="volume_id",
        choices=[volume_view(ctx, volume) for volume in usable],
    )


async def list_entries(
    ctx: ToolContext, cluster_id: str, volume_id: str | None = None, timeout: float = 30
) -> list[dict[str, Any]]:
    params = {"cluster_id": cluster_id, **({"storage_volume_id": volume_id} if volume_id else {})}
    return _items(await ctx.api.request("GET", "/model-cache/entries", params=params, timeout=timeout))


async def get_entry(ctx: ToolContext, entry_id: str) -> dict[str, Any] | None:
    try:
        return await ctx.api.request("GET", f"/model-cache/entries/{entry_id}", timeout=30)
    except LensApiError as error:
        if error.status == 404:
            return None
        raise


def entry_repo(entry: dict[str, Any]) -> str:
    return ((entry.get("source") or {}).get("huggingface") or {}).get("repoId") or ""


def entry_revision(entry: dict[str, Any]) -> str:
    return ((entry.get("source") or {}).get("huggingface") or {}).get("revision") or DEFAULT_REVISION


def entries_for_model(entries: list[dict[str, Any]], model: str, revision: str | None) -> list[dict[str, Any]]:
    wanted = (revision or DEFAULT_REVISION).lower()
    return [
        entry
        for entry in entries
        if entry_repo(entry).lower() == model.lower() and entry_revision(entry).lower() == wanted
    ]


def entry_ownership(ctx: ToolContext, entry: dict[str, Any]) -> tuple[str, Resource | None]:
    row = ctx.ledger.find_by_execution(entry.get("id") or "")
    if row is None or row.kind != "model_cache":
        return "unknown", None
    return ("owned" if row.owner == ctx.owner else "shared"), (row if row.owner == ctx.owner else None)


def node_progress(entry: dict[str, Any]) -> dict[str, int]:
    nodes = entry.get("nodeProgress") or []
    return {"ready": sum(1 for node in nodes if node.get("status") == "ready"), "total": len(nodes)}


def entry_view(ctx: ToolContext, entry: dict[str, Any]) -> dict[str, Any]:
    ownership, row = entry_ownership(ctx, entry)
    view: dict[str, Any] = {
        "lens_entry_id": entry.get("id"),
        "status": entry.get("status"),
        "volume_id": entry.get("storageVolumeId"),
        "ownership": ownership,
        "nodes": node_progress(entry),
    }
    if row:
        view["model_cache_id"] = row.id
    if entry.get("sizeBytes"):
        view["size_bytes"] = entry["sizeBytes"]
    failures = [entry.get("failureDetail")] + [node.get("failureDetail") for node in entry.get("nodeProgress") or []]
    failures = [str(text)[:300] for text in failures if text]
    if failures:
        view["untrusted_text"] = {"failure_detail": failures[0]}
    return view


def aggregate(ctx: ToolContext, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Cache state of one model across the matching entries: the best status plus every entry."""
    if not entries:
        return {"status": "absent", "entries": []}
    views = [entry_view(ctx, entry) for entry in entries]
    views.sort(key=lambda view: _STATUS_ORDER.index(view["status"]) if view["status"] in _STATUS_ORDER else 99)
    return {"status": views[0]["status"], "entries": views}


async def entry_create(ctx: ToolContext, cluster_id: str, volume_id: str, model: str, revision: str) -> dict[str, Any]:
    return await ctx.api.request(
        "POST",
        "/model-cache/entries",
        body={
            "clusterId": cluster_id,
            "storageVolumeId": volume_id,
            "source": {"kind": "huggingface", "huggingface": {"repoId": model, "revision": revision}},
            "force": False,
        },
        timeout=60,
    )


async def entry_retry(ctx: ToolContext, entry_id: str) -> dict[str, Any]:
    return await ctx.api.request("POST", f"/model-cache/entries/{entry_id}/retry", timeout=60)


# --- the ``model_cache`` kind behind list_resources / delete_resources ---------------------------------------


async def _delete(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    if not cascade:
        dependents = await _dependents(ctx, row)
        if dependents:
            raise ToolError("in_use", "Deployments still use this model cache.", dependents=dependents, field="ids")
    if not row.lens_execution_id:
        return
    try:
        await ctx.api.request("DELETE", f"/model-cache/entries/{row.lens_execution_id}", timeout=60)
    except LensApiError as error:
        if error.status != 404:
            raise


async def _describe(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    entry = await get_entry(ctx, row.lens_execution_id) if row.lens_execution_id else None
    if entry is None:
        return {"live_status": "missing", "model": row.meta.get("model")}
    view = entry_view(ctx, entry)
    return {
        "live_status": view["status"],
        "model": entry_repo(entry),
        "revision": entry_revision(entry),
        "volume_id": view["volume_id"],
        "nodes": view["nodes"],
        **({"size_bytes": view["size_bytes"]} if "size_bytes" in view else {}),
        **({"untrusted_text": view["untrusted_text"]} if "untrusted_text" in view else {}),
    }


async def _dependents(ctx: ToolContext, row: Resource) -> list[dict[str, Any]]:
    model = row.meta.get("model")
    if not row.cluster_id or not model:
        return []
    volume = row.meta.get("volume_id")
    try:
        payload = await ctx.api.request(
            "GET", "/deployments/executions", params={"cluster_id": row.cluster_id, "limit": 500}, timeout=30
        )
    except LensApiError as error:
        raise upstream_error(error) from error
    dependents = []
    for item in _items(payload):
        if item.get("status") not in _ACTIVE_DEPLOYMENT or str(item.get("model", "")).lower() != model.lower():
            continue
        if volume and item.get("storage_volume_id") != volume:
            continue
        owned = ctx.ledger.find_by_execution(item.get("execution_id") or "")
        dependents.append(
            {
                "id": owned.id if owned and owned.kind == "deployment" else item.get("execution_id"),
                "kind": "deployment",
                "name": item.get("name"),
                "reason": "serves this model from the cache volume",
            }
        )
    return dependents


async def _list_others(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    if not cluster_id:
        return []
    entries = await list_entries(ctx, cluster_id)
    others = []
    for entry in sorted(entries, key=lambda item: item.get("createdAt") or "", reverse=True):
        ownership, _ = entry_ownership(ctx, entry)
        if ownership == "owned":
            continue
        others.append(
            {
                "id": entry.get("id"),
                "lens_id": entry.get("id"),
                "kind": "model_cache",
                "name": f"{entry_repo(entry)}@{entry_revision(entry)}",
                "ownership": ownership,
                "cluster_id": entry.get("clusterId"),
                "status": entry.get("status"),
                "volume_id": entry.get("storageVolumeId"),
            }
        )
    return others[:50]


KIND_HANDLERS = {"delete": _delete, "describe": _describe, "dependents": _dependents, "list_others": _list_others}
