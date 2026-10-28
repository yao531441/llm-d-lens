"""Lens storage-volume REST calls, views, ownership checks and the storage_volume resource kind."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .kinds import register_kind
from .ledger import Resource
from .lens_api import LensApiError
from .reconcile_rules import CLAIM_WINDOW_SECONDS, claim_expired, may_claim, row_age

VOLUMES = "/storage/volumes"
ENTRIES = "/model-cache/entries"
_CAPACITY = re.compile(r"^(\d+)(Mi|Gi|Ti)$")
_GIB = {"Mi": 1 / 1024, "Gi": 1, "Ti": 1024}


@dataclass(frozen=True)
class VolumeRef:
    row: Resource | None  # ledger record when the caller created the volume
    volume_id: str
    foreign_owner: bool = False

    @property
    def owned(self) -> bool:
        return self.row is not None and self.row.state in ("active", "deleting")


def is_definite_failure(error: BaseException) -> bool:
    """True when Lens clearly refused the request, so nothing was created."""
    if isinstance(error, ToolError):
        return True
    return isinstance(error, LensApiError) and 400 <= error.status < 500 and error.status not in (408, 429)


def outcome_unknown(row: Resource, error: BaseException) -> ToolError:
    cause = f"HTTP {error.status}" if isinstance(error, LensApiError) else type(error).__name__
    return ToolError(
        "upstream_unavailable",
        f"Lens did not confirm whether volume '{row.name}' was created ({cause}); it may exist.",
        hint=f"Wait a moment, then call get_storage_volume(volume_id='{row.id}'); it reconciles by name. "
        "Do not create the same name again before that.",
        volume_id=row.id,
    )


def matches_request(volume: dict[str, Any], request: Any) -> bool:
    if not isinstance(request, dict):
        return False
    local = volume.get("localDisk") or {}
    nfs = volume.get("nfs") or {}
    return (
        volume.get("kind") == request.get("kind")
        and volume.get("capacity") == request.get("capacity")
        and (local.get("hostPath") or nfs.get("path")) == request.get("path")
        and (nfs.get("server") or None) == request.get("server")
    )


async def reconcile_volume(ctx: ToolContext, row: Resource) -> Resource:
    """Bind an unconfirmed ledger row to the Lens volume with its name on the same cluster, if Lens has one."""
    if row.lens_execution_id or row.state not in ("creating", "deleting") or claim_expired(row):
        return row
    try:
        data = await ctx.api.request("GET", VOLUMES, params={"cluster_id": row.cluster_id})
    except Exception:  # noqa: BLE001 - still unknown; the caller reports it
        return row
    for volume in data.get("items") or []:
        # volumes carry no owner, so the claim needs the name, the creation window and the exact spec we requested
        same = str(volume.get("name", "")).lower() == (row.name or "").lower()
        if same and matches_request(volume, row.meta.get("request")) and may_claim(ctx, row, volume):
            ctx.ledger.bind_execution(row.id, volume["id"])
            if row.state == "creating":
                ctx.ledger.set_state(row.id, "active")
            return ctx.ledger.get(row.id) or row
    return row


def pending_in_use(row: Resource) -> ToolError:
    wait = max(1, round(CLAIM_WINDOW_SECONDS - row_age(row)))
    return ToolError(
        "in_use",
        f"The creation of volume '{row.name}' is still unconfirmed and may yet complete in Lens.",
        hint=f"Call get_storage_volume(volume_id='{row.id}') to reconcile it; if nothing appears, "
        f"delete_resources will clear the record in about {wait} seconds.",
    )


def volume_not_found(volume_id: str) -> ToolError:
    return ToolError(
        "not_found",
        f"No storage volume '{volume_id}' is visible to you.",
        field="volume_id",
        hint="Call list_resources(kinds=['storage_volume'], owner='all') to see valid ids.",
    )


async def resolve_volume(ctx: ToolContext, volume_id: str) -> VolumeRef:
    """Accepts a ledger id (``vol-...``) or a raw Lens id (``storage-...``); ownership comes only from the ledger."""
    if volume_id.startswith("vol-"):
        row = ctx.ledger.get(volume_id)
        if row is None or row.kind != "storage_volume" or row.state == "deleted":
            raise volume_not_found(volume_id)
        if not row.lens_execution_id and row.owner == ctx.owner:
            row = await reconcile_volume(ctx, row)
        if not row.lens_execution_id:
            raise ToolError(
                "conflict_state",
                f"{volume_id} was being created but Lens has not confirmed it, so it may not exist.",
                field="volume_id",
                hint=(
                    "Retry shortly."
                    if not claim_expired(row)
                    else f"No matching volume appeared within {CLAIM_WINDOW_SECONDS // 60} minutes; "
                    "remove the record with delete_resources(ids=['" + volume_id + "'])."
                ),
            )
        if row.owner != ctx.owner:
            return VolumeRef(None, row.lens_execution_id, foreign_owner=True)
        return VolumeRef(row, row.lens_execution_id)
    row = ctx.ledger.find_by_execution(volume_id)
    if row is not None and (row.kind != "storage_volume" or row.owner != ctx.owner):
        row = None
    return VolumeRef(row, volume_id)


def require_owned(ref: VolumeRef, volume_id: str) -> Resource:
    if ref.row is not None and ref.owned:
        return ref.row
    if ref.foreign_owner:
        raise ToolError(
            "not_owner",
            f"{volume_id} belongs to another user. Agents cannot change other people's volumes.",
            field="volume_id",
            hint="Use list_resources(kinds=['storage_volume']) to see the volumes you own.",
        )
    raise ToolError(
        "owner_unknown",
        f"{volume_id} was not created through this server, so its owner cannot be verified.",
        field="volume_id",
        hint="Only volumes created with create_storage_volume can be changed here. Ask the owner or an administrator.",
    )


async def load_volume(ctx: ToolContext, volume_id: str) -> dict[str, Any]:
    try:
        return await ctx.api.request("GET", f"{VOLUMES}/{volume_id}")
    except LensApiError as error:
        if error.status == 404:
            raise volume_not_found(volume_id) from error
        raise


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


def model_rows(ctx: ToolContext, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        huggingface = (entry.get("source") or {}).get("huggingface") or {}
        progress = entry.get("nodeProgress") or []
        known = ctx.ledger.find_by_execution(entry["id"])
        rows.append(
            {
                "model": huggingface.get("repoId") or entry.get("name") or entry["id"],
                "revision": huggingface.get("revision"),
                "status": entry.get("status"),
                "size_gib": round(entry["sizeBytes"] / 2**30, 2) if entry.get("sizeBytes") else None,
                "model_cache_id": known.id if known else entry["id"],
                "managed_by_lens": known is not None,
                "nodes_ready": sorted(p["node"] for p in progress if p.get("status") == "ready"),
                "nodes_pending": sorted(p["node"] for p in progress if p.get("status") != "ready"),
                "pending_sync_nodes": entry.get("pendingSyncNodes") or [],
            }
        )
    return rows


def volume_view(
    ctx: ToolContext,
    ref: VolumeRef,
    volume: dict[str, Any],
    *,
    nodes: list[dict[str, Any]] | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "volume_id": ref.row.id if ref.row else volume["id"],
        "lens_id": volume["id"],
        "name": volume["name"],
        "is_mine": ref.owned,
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
        view["models"] = model_rows(ctx, entries)
    return view


def _location(volume: dict[str, Any]) -> str | None:
    if volume.get("localDisk"):
        return f"hostPath {volume['localDisk']['hostPath']}"
    if volume.get("nfs"):
        return f"nfs {volume['nfs']['server']}:{volume['nfs']['path']}"
    return None


# --- storage_volume kind: used by list_resources and delete_resources --------------------------------


async def describe(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    row = await reconcile_volume(ctx, row)
    if not row.lens_execution_id:
        return {"live_status": "unconfirmed"}
    try:
        volume = await load_volume(ctx, row.lens_execution_id or "")
    except ToolError:
        return {"live_status": "missing"}
    return {
        "live_status": volume["status"],
        "kind": volume["kind"],
        "capacity_gib": capacity_gib(volume.get("capacity")),
        "in_use_count": volume.get("inUseCount", 0),
    }


async def dependents(ctx: ToolContext, row: Resource) -> list[dict[str, Any]]:
    row = await reconcile_volume(ctx, row)
    volume_id = row.lens_execution_id
    if not volume_id:
        return []
    try:
        volume = await load_volume(ctx, volume_id)
        entries = await load_entries(ctx, volume_id)
    except ToolError:
        return []
    items = []
    if volume.get("inUseCount"):
        items.append(
            {
                "id": volume_id,
                "kind": "deployment",
                "name": f"{volume['inUseCount']} active deployment(s)",
                "reason": "they mount this volume and would lose their model files; Lens refuses the delete",
            }
        )
    for item in model_rows(ctx, entries):
        items.append(
            {
                "id": item["model_cache_id"],
                "kind": "model_cache",
                "name": item["model"],
                "reason": "the cache record is removed with the volume; model files stay on disk",
            }
        )
    return items


async def delete(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    row = await reconcile_volume(ctx, row)
    volume_id = row.lens_execution_id
    if not volume_id:
        if not claim_expired(row):
            raise pending_in_use(row)
        return
    try:
        # Keep the files: deleting the Lens record must not erase models that may be shared or hand-copied.
        await ctx.api.request("DELETE", f"{VOLUMES}/{volume_id}", params={"keep_model_files": "true"}, timeout=120)
    except LensApiError as error:
        if error.status == 404:
            return
        if error.status == 409:
            raise ToolError(
                "in_use",
                "Deployments still use this volume.",
                hint="Delete or move those deployments first (delete_resources or stop_deployment), then retry.",
            ) from error
        raise
    for cache in ctx.ledger.list_resources(owner=ctx.owner, kind="model_cache"):
        if cache.meta.get("storage_volume_id") == volume_id:
            ctx.ledger.set_state(cache.id, "deleted")


async def list_others(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", VOLUMES, params={"cluster_id": cluster_id} if cluster_id else None)
    entries = []
    for volume in data.get("items") or []:
        known = ctx.ledger.find_by_execution(volume["id"])
        entries.append(
            {
                "id": volume["id"],
                "lens_id": volume["id"],
                "kind": "storage_volume",
                "name": volume["name"],
                "ownership": "shared" if known is not None else "unknown",
                "state": volume.get("status"),
                "cluster_id": volume.get("clusterId"),
            }
        )
    return entries


register_kind("storage_volume", delete=delete, describe=describe, dependents=dependents, list_others=list_others)
