"""Storage profile: create, inspect and scan model-cache volumes."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..context import ToolContext
from ..errors import ToolError
from ..lens_api import LensApiError
from ..registry import tool
from ..storage_ops import (
    VOLUMES,
    VolumeRef,
    load_entries,
    load_volume,
    require_owned,
    resolve_volume,
    volume_view,
)
from ..storage_paths import check_host_path

CREATED_VIA = "lens-mcp-v2"


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def _nodes(ctx: ToolContext, cluster_id: str) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", "/storage/nodes", params={"cluster_id": cluster_id})
    return data.get("items") or []


# --- create ----------------------------------------------------------------------------------------------


class CreateVolumeArgs(Args):
    cluster_id: str = Field(description="Cluster id from list_clusters.")
    name: str = Field(
        pattern=r"^[a-z0-9]([a-z0-9-]{0,40}[a-z0-9])?$",
        description="Volume name; lowercase letters, digits and hyphens. Unique per cluster.",
    )
    kind: Literal["local-disk", "nfs"] = Field(
        description="local-disk=the same hostPath directory on every node (fast, models must be present on each "
        "node); nfs=a shared NFS export. dynamic-pvc is not offered."
    )
    path: str = Field(
        pattern=r"^/[A-Za-z0-9._/-]+$",
        description="local-disk: directory on the nodes, which must be under an allowed prefix (the error lists "
        "them). nfs: export path.",
    )
    nfs_server: str | None = Field(default=None, description="Required for kind='nfs': server host or IP.")
    capacity_gib: int = Field(ge=10, le=20000, description="Declared capacity in GiB.")
    ttl_hours: int = Field(default=24, ge=1, le=720, description="Recorded lifetime in hours; nothing reaps it yet.")
    client_request_id: str | None = Field(default=None, description="Retry key: repeating it returns the first result.")

    @model_validator(mode="after")
    def _nfs_needs_server(self):
        if self.kind == "nfs" and not self.nfs_server:
            raise ValueError("nfs_server is required when kind is 'nfs'")
        if self.kind == "local-disk" and self.nfs_server:
            raise ValueError("nfs_server is only valid when kind is 'nfs'")
        return self


@tool(
    "create_storage_volume",
    "[S] Register a model-cache storage volume (a node-local hostPath or an NFS export) in Lens for caching models. "
    "hostPath is only allowed under prefixes the operator configured; otherwise the call fails and lists them. "
    "Lens volumes are node-agnostic, so you cannot pick nodes. Check progress with get_storage_volume; "
    "models already on the volume are imported by Lens once it is ready.\n"
    "Side effects: creates a Lens volume record and PV/PVC; resources: storage capacity only; "
    "duration: seconds to a minute; reversible: delete_resources(ids=[vol-...]).",
    CreateVolumeArgs,
    read_only=False,
    idempotent=True,
    profile="storage",
)
async def create_storage_volume(ctx: ToolContext, args: CreateVolumeArgs) -> dict[str, Any]:
    path = check_host_path(args.path) if args.kind == "local-disk" else args.path
    existing = (await ctx.api.request("GET", VOLUMES, params={"cluster_id": args.cluster_id})).get("items") or []
    if any(item["name"].lower() == args.name.lower() for item in existing):
        raise ToolError(
            "already_exists",
            f"A storage volume named '{args.name}' already exists on this cluster.",
            field="name",
            hint="Pick another name; list_resources(kinds=['storage_volume'], owner='all') shows existing volumes.",
        )
    spec = (
        {"localDisk": {"hostPath": path}}
        if args.kind == "local-disk"
        else {"nfs": {"server": args.nfs_server, "path": path}}
    )
    row = ctx.ledger.add_resource(
        kind="storage_volume",
        owner=ctx.owner,
        cluster_id=args.cluster_id,
        name=args.name,
        state="creating",
        meta={"created_via": CREATED_VIA, "expires_at": time.time() + args.ttl_hours * 3600},
    )
    try:
        volume = await ctx.api.request(
            "POST",
            VOLUMES,
            body={
                "clusterId": args.cluster_id,
                "name": args.name,
                "kind": args.kind,
                "capacity": f"{args.capacity_gib}Gi",
                "readOnly": False,
                "purposes": ["model-cache"],
                **spec,
            },
            timeout=120,
        )
    except LensApiError as error:
        ctx.ledger.set_state(row.id, "deleted")
        if error.status == 409:
            raise ToolError("already_exists", error.message(), field="name") from error
        raise
    except BaseException:
        ctx.ledger.set_state(row.id, "deleted")
        raise
    ctx.ledger.bind_execution(row.id, volume["id"])
    ctx.ledger.set_state(row.id, "active")
    view = volume_view(ctx, VolumeRef(ctx.ledger.get(row.id), volume["id"]), volume)
    hint = (
        "The same hostPath is used on every node; the directory must exist on each of them. "
        if args.kind == "local-disk"
        else ""
    )
    return {
        **view,
        "next": f"{hint}Poll get_storage_volume(volume_id='{row.id}', wait_seconds=20) until status is ready.",
    }


# --- get -------------------------------------------------------------------------------------------------


class GetVolumeArgs(Args):
    volume_id: str = Field(description="Volume id (vol-...) or Lens id (storage-...) from list_resources.")
    include: list[Literal["nodes", "models", "k8s_status"]] = Field(
        default_factory=lambda: ["nodes", "models"],
        description="nodes=node coverage and drift; models=models on the volume; k8s_status=PV/PVC state (debugging).",
    )
    wait_seconds: int = Field(default=0, ge=0, le=25, description="Wait up to this long while status is 'pending'.")


@tool(
    "get_storage_volume",
    "[R] Show a storage volume: status, capacity, node coverage and drift, and the models on it (with whether Lens "
    "manages them). Works for volumes of others in a read-only view. Use list_resources to find volumes and "
    "scan_storage_volume to refresh models and nodes.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    GetVolumeArgs,
    read_only=True,
    profile="storage",
)
async def get_storage_volume(ctx: ToolContext, args: GetVolumeArgs) -> dict[str, Any]:
    ref = await resolve_volume(ctx, args.volume_id)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        volume = await load_volume(ctx, ref.volume_id)
        if volume["status"] != "pending" or time.monotonic() >= deadline:
            break
        await asyncio.sleep(2)
    nodes = await _nodes(ctx, volume["clusterId"]) if "nodes" in args.include else None
    entries = await load_entries(ctx, ref.volume_id) if "models" in args.include else None
    view = volume_view(ctx, ref, volume, nodes=nodes, entries=entries)
    if "k8s_status" in args.include:
        data = await ctx.api.request("GET", "/storage/volume-resource-status", params={"volume_id": ref.volume_id})
        view["k8s_status"] = data.get("items") or []
    return view


# --- scan ------------------------------------------------------------------------------------------------


class ScanVolumeArgs(Args):
    volume_id: str = Field(description="Your own volume id (vol-...).")
    register_models: bool = Field(
        default=True,
        description="Register models found on the volume as cache entries. Lens cannot scan without registering, "
        "so false sends no scan and only reports the current state.",
    )
    confirm_nodes: bool = Field(
        default=False,
        description="After validating that every new node is ready and holds every cached model, accept the current "
        "nodes as the volume's baseline. Refused with details when validation fails.",
    )
    wait_seconds: int = Field(default=5, ge=0, le=25, description="Time to let the background scan register models.")


async def _validate_nodes(
    ctx: ToolContext, volume: dict[str, Any], entries: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    drift = volume.get("nodesAdded") or []
    live = {n["name"]: n for n in await _nodes(ctx, volume["clusterId"])}
    problems = []
    for node in drift:
        state = live.get(node)
        if state is None:
            problems.append(f"{node}: node no longer exists in the cluster")
        elif not state.get("ready", True) or not state.get("schedulable", True):
            problems.append(f"{node}: node is not ready or not schedulable")
    for entry in entries:
        if entry.get("status") != "ready":
            continue
        ready = {p["node"] for p in entry.get("nodeProgress") or [] if p.get("status") == "ready"}
        for node in drift:
            if node not in ready:
                name = ((entry.get("source") or {}).get("huggingface") or {}).get("repoId") or entry["id"]
                problems.append(f"{node}: model {name} is not synced to this node")
    return drift, problems


@tool(
    "scan_storage_volume",
    "[W] Re-scan your own volume for models copied onto it outside Lens (registering them as cache entries), and "
    "optionally confirm new nodes after checking they are ready and hold every cached model. Nothing is deleted. "
    "Use get_storage_volume to inspect without changing anything.\n"
    "Side effects: registers cache records and may re-baseline the node list; resources: none; "
    "duration: seconds; reversible: registered cache records can be deleted with delete_resources.",
    ScanVolumeArgs,
    read_only=False,
    idempotent=True,
    profile="storage",
)
async def scan_storage_volume(ctx: ToolContext, args: ScanVolumeArgs) -> dict[str, Any]:
    ref = await resolve_volume(ctx, args.volume_id)
    require_owned(ref, args.volume_id)
    volume = await load_volume(ctx, ref.volume_id)
    if volume["status"] != "ready":
        raise ToolError(
            "conflict_state",
            f"The volume is {volume['status']}, not ready.",
            field="volume_id",
            hint="Wait for get_storage_volume to report status=ready, then scan again.",
        )
    before = {entry["id"] for entry in await load_entries(ctx, ref.volume_id)}
    scan: dict[str, Any] = {"requested": args.register_models}
    if args.register_models:
        await ctx.api.request("POST", f"{VOLUMES}/{ref.volume_id}/scan-models", timeout=120)
        await asyncio.sleep(args.wait_seconds)
    entries = await load_entries(ctx, ref.volume_id)
    if args.register_models:
        scan["new_models"] = _register_new(ctx, ref, volume, entries, before)
        scan["note"] = "The scan runs in the background; call again or use get_storage_volume if models are missing."
    else:
        scan["note"] = "No scan was sent: Lens always registers what it finds, so a scan needs register_models=true."
    result: dict[str, Any] = {"scan": scan}
    if args.confirm_nodes:
        result["nodes_confirmation"] = await _confirm_nodes(ctx, ref, volume, entries)
        volume = await load_volume(ctx, ref.volume_id)
    nodes = await _nodes(ctx, volume["clusterId"])
    result.update(volume_view(ctx, ref, volume, nodes=nodes, entries=entries))
    result["next"] = "Review models and nodes; use get_storage_volume for details."
    return result


def _register_new(
    ctx: ToolContext, ref: VolumeRef, volume: dict[str, Any], entries: list[dict[str, Any]], before: set[str]
) -> list[dict[str, Any]]:
    registered = []
    for entry in entries:
        if entry["id"] in before or ctx.ledger.find_by_execution(entry["id"]) is not None:
            continue
        name = ((entry.get("source") or {}).get("huggingface") or {}).get("repoId") or entry["id"]
        row = ctx.ledger.add_resource(
            kind="model_cache",
            owner=ctx.owner,
            cluster_id=volume.get("clusterId"),
            name=name,
            lens_execution_id=entry["id"],
            meta={"created_via": CREATED_VIA, "source": "scan", "storage_volume_id": ref.volume_id},
        )
        registered.append({"model": name, "model_cache_id": row.id})
    return registered


async def _confirm_nodes(
    ctx: ToolContext, ref: VolumeRef, volume: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    if volume["kind"] != "local-disk":
        return {"confirmed": False, "reason": "Only local-disk volumes have a node baseline."}
    drift, problems = await _validate_nodes(ctx, volume, entries)
    if not drift:
        return {"confirmed": False, "reason": "No new nodes to confirm."}
    if problems:
        return {
            "confirmed": False,
            "new_nodes": drift,
            "problems": problems,
            "hint": "Copy the model files to the new nodes (or fix the nodes), scan again, then confirm.",
        }
    await ctx.api.request("POST", f"{VOLUMES}/{ref.volume_id}/acknowledge-nodes", timeout=60)
    return {"confirmed": True, "new_nodes": drift}
