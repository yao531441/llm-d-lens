"""Lens storage-volume REST calls, views and the storage_volume resource kind.

Lens records no owner for volumes, so Lens permissions alone decide who may change one.
"""

from __future__ import annotations

import re
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .kinds import register_kind
from .lens_api import LensApiError
from .path_ids import check_id

VOLUMES = "/storage/volumes"
ENTRIES = "/model-cache/entries"
_CAPACITY = re.compile(r"^(\d+)(Mi|Gi|Ti)$")
_GIB = {"Mi": 1 / 1024, "Gi": 1, "Ti": 1024}


def volume_not_found(volume_id: str) -> ToolError:
    return ToolError(
        "not_found",
        f"No storage volume '{volume_id}' is visible to you.",
        field="volume_id",
        hint="Call resource_list(kinds=['storage_volume'], owner='all') to see valid ids.",
    )


async def get_volume(ctx: ToolContext, volume_id: str) -> dict[str, Any] | None:
    check_id(volume_id, "volume_id")
    try:
        return await ctx.api.request("GET", f"{VOLUMES}/{volume_id}")
    except LensApiError as error:
        if error.status == 404:
            return None
        raise


async def load_volume(ctx: ToolContext, volume_id: str) -> dict[str, Any]:
    volume = await get_volume(ctx, volume_id)
    if volume is None:
        raise volume_not_found(volume_id)
    return volume


async def load_entries(ctx: ToolContext, volume_id: str) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", ENTRIES, params={"storage_volume_id": volume_id})
    return data.get("items") or []


def capacity_gib(capacity: str | None) -> int | None:
    match = _CAPACITY.match(capacity or "")
    return round(int(match.group(1)) * _GIB[match.group(2)]) if match else None


def node_rows(volume: dict[str, Any], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-node coverage of a local-disk volume: the baseline in knownNodes versus the nodes that exist now."""
    known = set(volume.get("knownNodes") or [])
    pending = set(volume.get("nodesAdded") or [])
    live = {n["name"]: n for n in nodes}
    rows = []
    for name in sorted(known | pending | set(live)):
        node = live.get(name)
        if node is None:
            status = "missing"
        elif not node.get("ready", True) or not node.get("schedulable", True):
            status = "unavailable"
        elif name in pending or name not in known:
            status = "pending_ack"
        else:
            status = "ready"
        rows.append({"node": name, "status": status, "in_baseline": name in known})
    return rows


def model_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        huggingface = (entry.get("source") or {}).get("huggingface") or {}
        progress = entry.get("nodeProgress") or []
        rows.append(
            {
                "model": huggingface.get("repoId") or entry.get("name") or entry["id"],
                "revision": huggingface.get("revision"),
                "status": entry.get("status"),
                "size_gib": round(entry["sizeBytes"] / 2**30, 2) if entry.get("sizeBytes") else None,
                "model_cache_id": entry["id"],
                "nodes_ready": sorted(p["node"] for p in progress if p.get("status") == "ready"),
                "nodes_pending": sorted(p["node"] for p in progress if p.get("status") != "ready"),
                "pending_sync_nodes": entry.get("pendingSyncNodes") or [],
            }
        )
    return rows


def volume_view(
    volume: dict[str, Any],
    *,
    nodes: list[dict[str, Any]] | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "volume_id": volume["id"],
        "name": volume["name"],
        "cluster_id": volume.get("clusterId"),
        "kind": volume["kind"],
        "status": volume["status"],
        "capacity_gib": capacity_gib(volume.get("capacity")),
        "used_gib": round(volume["usedBytes"] / 2**30, 2) if volume.get("usedBytes") is not None else None,
        "read_only": bool(volume.get("readOnly")),
        "location": _location(volume),
        "in_use_count": volume.get("inUseCount", 0),
        "model_cache_count": volume.get("modelCacheCount"),
    }
    if volume.get("failureDetail"):
        view["untrusted_text"] = {"failure_detail": str(volume["failureDetail"])[:500]}
    if volume["kind"] == "local-disk":
        view["node_drift"] = volume.get("nodesAdded") or []
    if nodes is not None:
        view["nodes"] = node_rows(volume, nodes) if volume["kind"] == "local-disk" else "shared by all nodes"
    if entries is not None:
        view["models"] = model_rows(entries)
    return view


def _location(volume: dict[str, Any]) -> str | None:
    if volume.get("localDisk"):
        return f"hostPath {volume['localDisk']['hostPath']}"
    if volume.get("nfs"):
        return f"nfs {volume['nfs']['server']}:{volume['nfs']['path']}"
    return None


# --- storage_volume kind: used by resource_list and resource_delete --------------------------------


def _entry(volume: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": volume["id"],
        "kind": "storage_volume",
        "name": volume["name"],
        "status": volume.get("status"),
        "ownership": "unrecorded",
        "cluster_id": volume.get("clusterId"),
        "created_at": volume.get("createdAt"),
        "volume_kind": volume["kind"],
        "capacity_gib": capacity_gib(volume.get("capacity")),
        "in_use_count": volume.get("inUseCount", 0),
    }


async def list_volume_entries(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", VOLUMES, params={"cluster_id": cluster_id} if cluster_id else None)
    return [_entry(volume) for volume in data.get("items") or []]


async def get_volume_entry(ctx: ToolContext, volume_id: str) -> dict[str, Any] | None:
    volume = await get_volume(ctx, volume_id)
    return _entry(volume) if volume else None


async def delete_volume(ctx: ToolContext, volume_id: str) -> list[dict[str, str]]:
    """Delete the Lens record and keep the files: they may be shared or copied by hand."""
    caches = [
        {"kind": "model_cache", "id": row["model_cache_id"]} for row in model_rows(await load_entries(ctx, volume_id))
    ]
    try:
        await ctx.api.request("DELETE", f"{VOLUMES}/{volume_id}", params={"keep_model_files": "true"}, timeout=120)
    except LensApiError as error:
        if error.status == 404:
            return []
        if error.status == 409:
            raise ToolError(
                "in_use",
                "Deployments still use this volume.",
                hint="Delete or move those deployments first (resource_delete or deployment_stop), then retry.",
            ) from error
        raise
    return caches


register_kind("storage_volume", list=list_volume_entries, get=get_volume_entry, delete=delete_volume)
