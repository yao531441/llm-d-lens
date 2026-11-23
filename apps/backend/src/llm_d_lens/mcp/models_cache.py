"""Model-cache and storage-volume views shared by the model tools, ``cluster_get`` and the ``model_cache`` kind.

Lens records no owner for cache entries or storage volumes, so they are ``unrecorded`` in the resource tools and
Lens permissions alone decide who may change them.
"""

from __future__ import annotations

from typing import Any

from .context import ToolContext
from .errors import ToolError
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


def volume_view(volume: dict[str, Any]) -> dict[str, Any]:
    return {
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


def create_volume_advice(ctx: ToolContext) -> str:
    """How to get a model-cache volume, depending on whether this connection may create one."""
    if "storage" in ctx.scopes:
        return (
            "Register one with storage_volume_create (it always creates a model-cache volume; cluster_id, name, kind, "
            "path and capacity_gib are required), wait until it is ready, then retry. If Lens refuses, ask the Lens "
            "administrator."
        )
    return (
        "storage_volume_create is not available on this connection (it is in the 'storage' profile). Ask the Lens "
        "administrator to register a model-cache volume in the Lens UI (Storage page) or to enable that profile."
    )


def no_usable_volume(
    ctx: ToolContext, cluster_id: str, volumes: list[dict[str, Any]], also: str = "", field: str = "volume_id"
) -> ToolError:
    existing = [
        f"{volume.get('id')} ({volume.get('status')}, purposes {volume.get('purposes') or []})" for volume in volumes
    ]
    state = f"Volumes on it: {'; '.join(existing)}." if existing else "It has no storage volumes registered."
    return ToolError(
        "prerequisite_missing",
        f"Cluster {cluster_id} has no ready model-cache volume. {state}",
        field=field,
        hint=create_volume_advice(ctx) + also,
    )


async def resolve_volume(
    ctx: ToolContext, cluster_id: str, volume_id: str | None, field: str = "volume_id"
) -> dict[str, Any]:
    """Pick the ready model-cache volume to download into: the given Lens id, or the only choice."""
    volumes = await list_volumes(ctx, cluster_id)
    usable = [volume for volume in volumes if is_model_cache_volume(volume) and volume.get("status") == "ready"]
    if volume_id:
        for volume in volumes:
            if volume.get("id") == volume_id:
                if not is_model_cache_volume(volume):
                    raise ToolError(
                        "invalid_argument",
                        f"Volume {volume_id} is not a model-cache volume.",
                        field=field,
                        hint="Pick a volume with the model-cache purpose; cluster_get(include=['storage']) lists them.",
                    )
                if volume.get("status") != "ready":
                    raise ToolError(
                        "prerequisite_missing",
                        f"Volume {volume_id} is {volume.get('status')}, not ready.",
                        field=field,
                    )
                return volume
        raise ToolError(
            "not_found",
            f"No volume {volume_id} on cluster {cluster_id}.",
            field=field,
            hint="cluster_get(include=['storage']) lists the volumes registered on this cluster.",
        )
    if len(usable) == 1:
        return usable[0]
    if not usable:
        raise no_usable_volume(ctx, cluster_id, volumes, field=field)
    raise ToolError(
        "ambiguous_choice",
        f"Cluster {cluster_id} has {len(usable)} ready model-cache volumes; choose one.",
        field=field,
        choices=[volume_view(volume) for volume in usable],
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


def node_progress(entry: dict[str, Any]) -> dict[str, int]:
    nodes = entry.get("nodeProgress") or []
    return {"ready": sum(1 for node in nodes if node.get("status") == "ready"), "total": len(nodes)}


def entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        "model_cache_id": entry.get("id"),
        "status": entry.get("status"),
        "volume_id": entry.get("storageVolumeId"),
        "nodes": node_progress(entry),
    }
    if entry.get("sizeBytes"):
        view["size_bytes"] = entry["sizeBytes"]
    failures = [entry.get("failureDetail")] + [node.get("failureDetail") for node in entry.get("nodeProgress") or []]
    failures = [str(text)[:300] for text in failures if text]
    if failures:
        view["untrusted_text"] = {"failure_detail": failures[0]}
    return view


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Cache state of one model across the matching entries: the best status plus every entry."""
    if not entries:
        return {"status": "absent", "entries": []}
    views = [entry_view(entry) for entry in entries]
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


# --- the ``model_cache`` kind behind resource_list / resource_delete ---------------------------------------


async def scanned_clusters(ctx: ToolContext, cluster_id: str | None) -> list[str]:
    """The given cluster, or every cluster Lens manages."""
    if cluster_id:
        return [cluster_id]
    return [item["id"] for item in _items(await ctx.api.request("GET", "/cluster/clusters"))]


def _resource_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry.get("id"),
        "kind": "model_cache",
        "name": f"{entry_repo(entry)}@{entry_revision(entry)}",
        "status": entry.get("status"),
        "ownership": "unrecorded",
        "cluster_id": entry.get("clusterId"),
        "created_at": entry.get("createdAt"),
        "volume_id": entry.get("storageVolumeId"),
        "nodes": node_progress(entry),
    }


async def list_cache_entries(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    entries = []
    for cluster in await scanned_clusters(ctx, cluster_id):
        entries.extend(await list_entries(ctx, cluster))
    return [_resource_entry(entry) for entry in sorted(entries, key=lambda e: e.get("createdAt") or "", reverse=True)]


async def get_cache_entry(ctx: ToolContext, entry_id: str) -> dict[str, Any] | None:
    entry = await get_entry(ctx, entry_id)
    return _resource_entry(entry) if entry else None


async def delete_cache_entry(ctx: ToolContext, entry_id: str) -> list[dict[str, str]]:
    entry = await get_entry(ctx, entry_id)
    if entry is None:
        return []
    users = await _deployments_using(ctx, entry)
    if users:
        raise ToolError(
            "in_use",
            "Deployments still serve this model from the cache volume.",
            field="items",
            hint="Stop or delete them first.",
            dependents=users,
        )
    try:
        await ctx.api.request("DELETE", f"/model-cache/entries/{entry_id}", timeout=60)
    except LensApiError as error:
        if error.status != 404:
            raise
    return []


async def _deployments_using(ctx: ToolContext, entry: dict[str, Any]) -> list[dict[str, Any]]:
    model, cluster_id, volume = entry_repo(entry), entry.get("clusterId"), entry.get("storageVolumeId")
    if not cluster_id or not model:
        return []
    try:
        payload = await ctx.api.request(
            "GET", "/deployments/executions", params={"cluster_id": cluster_id, "limit": 500}, timeout=30
        )
    except LensApiError as error:
        raise upstream_error(error) from error
    return [
        {"kind": "deployment", "id": item.get("run_id") or item.get("execution_id"), "name": item.get("name")}
        for item in _items(payload)
        if item.get("status") in _ACTIVE_DEPLOYMENT
        and str(item.get("model", "")).lower() == model.lower()
        and (not volume or item.get("storage_volume_id") == volume)
    ]
