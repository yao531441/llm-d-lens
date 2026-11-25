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
    load_entries,
    load_volume,
    volume_view,
)
from ..storage_paths import check_host_path


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def _nodes(ctx: ToolContext, cluster_id: str) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", "/storage/nodes", params={"cluster_id": cluster_id})
    return data.get("items") or []


# --- create ----------------------------------------------------------------------------------------------


class CreateVolumeArgs(Args):
    cluster_id: str = Field(description="Cluster id from cluster_list.")
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

    @model_validator(mode="after")
    def _nfs_needs_server(self):
        if self.kind == "nfs" and not self.nfs_server:
            raise ValueError("nfs_server is required when kind is 'nfs'")
        if self.kind == "local-disk" and self.nfs_server:
            raise ValueError("nfs_server is only valid when kind is 'nfs'")
        return self


@tool(
    "storage_volume_create",
    "[S] Register a model-cache storage volume (a node-local hostPath or an NFS export) in Lens for caching models. "
    "hostPath is only allowed under prefixes the operator configured; otherwise the call fails and lists them. "
    "Lens volumes are node-agnostic, so you cannot pick nodes. Check progress with storage_volume_get; "
    "models already on the volume are imported by Lens once it is ready.\n"
    "Side effects: creates a Lens volume record and PV/PVC; resources: storage capacity only; "
    "duration: seconds to a minute; reversible: resource_delete.",
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
            hint="Pick another name; resource_list(kinds=['storage_volume'], owner='all') shows existing volumes.",
        )
    spec = (
        {"localDisk": {"hostPath": path}}
        if args.kind == "local-disk"
        else {"nfs": {"server": args.nfs_server, "path": path}}
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
        if error.status == 409:
            raise ToolError("already_exists", error.message(), field="name") from error
        raise
    view = volume_view(volume)
    hint = (
        "The same hostPath is used on every node; the directory must exist on each of them. "
        if args.kind == "local-disk"
        else ""
    )
    return {
        **view,
        "next": f"{hint}Poll storage_volume_get(volume_id='{volume['id']}', wait_seconds=20) until status is ready.",
    }


# --- get -------------------------------------------------------------------------------------------------


class GetVolumeArgs(Args):
    volume_id: str = Field(description="Volume id from resource_list or storage_volume_create.")
    include: list[Literal["nodes", "models", "k8s_status"]] = Field(
        default_factory=lambda: ["nodes", "models"],
        description="nodes=node coverage and drift; models=models on the volume; k8s_status=PV/PVC state (debugging).",
    )
    wait_seconds: int = Field(default=0, ge=0, le=25, description="Wait up to this long while status is 'pending'.")


@tool(
    "storage_volume_get",
    "[R] Show a storage volume: status, capacity, node coverage and drift, and the models on it (with whether Lens "
    "manages them). Works for volumes of others in a read-only view. Use resource_list to find volumes and "
    "storage_volume_scan to refresh models and nodes.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    GetVolumeArgs,
    read_only=True,
    profile="storage",
)
async def get_storage_volume(ctx: ToolContext, args: GetVolumeArgs) -> dict[str, Any]:
    deadline = time.monotonic() + args.wait_seconds
    while True:
        volume = await load_volume(ctx, args.volume_id)
        if volume["status"] != "pending" or time.monotonic() >= deadline:
            break
        await asyncio.sleep(2)
    nodes = await _nodes(ctx, volume["clusterId"]) if "nodes" in args.include else None
    entries = await load_entries(ctx, args.volume_id) if "models" in args.include else None
    view = volume_view(volume, nodes=nodes, entries=entries)
    if "k8s_status" in args.include:
        data = await ctx.api.request("GET", "/storage/volume-resource-status", params={"volume_id": args.volume_id})
        view["k8s_status"] = data.get("items") or []
    return view


# --- scan ------------------------------------------------------------------------------------------------


class ScanVolumeArgs(Args):
    volume_id: str = Field(description="Volume id from resource_list.")
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
    "storage_volume_scan",
    "[W] Re-scan a volume for models copied onto it outside Lens (registering them as cache entries), and "
    "optionally confirm new nodes after checking they are ready and hold every cached model. Nothing is deleted. "
    "Use storage_volume_get to inspect without changing anything.\n"
    "Side effects: registers cache records and may re-baseline the node list; resources: none; "
    "duration: seconds; reversible: registered cache records can be deleted with resource_delete.",
    ScanVolumeArgs,
    read_only=False,
    idempotent=True,
    profile="storage",
)
async def scan_storage_volume(ctx: ToolContext, args: ScanVolumeArgs) -> dict[str, Any]:
    volume = await load_volume(ctx, args.volume_id)
    if volume["status"] != "ready":
        raise ToolError(
            "conflict_state",
            f"The volume is {volume['status']}, not ready.",
            field="volume_id",
            hint="Wait for storage_volume_get to report status=ready, then scan again.",
        )
    before = {entry["id"] for entry in await load_entries(ctx, args.volume_id)}
    scan: dict[str, Any] = {"requested": args.register_models}
    if args.register_models:
        await ctx.api.request("POST", f"{VOLUMES}/{args.volume_id}/scan-models", timeout=120)
        await asyncio.sleep(args.wait_seconds)
    entries = await load_entries(ctx, args.volume_id)
    if args.register_models:
        scan["new_models"] = _new_models(entries, before)
        scan["note"] = "The scan runs in the background; call again or use storage_volume_get if models are missing."
    else:
        scan["note"] = "No scan was sent: Lens always registers what it finds, so a scan needs register_models=true."
    result: dict[str, Any] = {"scan": scan}
    if args.confirm_nodes:
        result["nodes_confirmation"] = await _confirm_nodes(ctx, volume, entries)
        volume = await load_volume(ctx, args.volume_id)
    nodes = await _nodes(ctx, volume["clusterId"])
    result.update(volume_view(volume, nodes=nodes, entries=entries))
    result["next"] = "Review models and nodes; use storage_volume_get for details."
    return result


def _new_models(entries: list[dict[str, Any]], before: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "model": ((entry.get("source") or {}).get("huggingface") or {}).get("repoId") or entry["id"],
            "model_cache_id": entry["id"],
        }
        for entry in entries
        if entry["id"] not in before
    ]


async def _confirm_nodes(ctx: ToolContext, volume: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
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
    await ctx.api.request("POST", f"{VOLUMES}/{volume['id']}/acknowledge-nodes", timeout=60)
    return {"confirmed": True, "new_nodes": drift}
