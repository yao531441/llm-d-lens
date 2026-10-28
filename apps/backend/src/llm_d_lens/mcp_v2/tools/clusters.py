"""Cluster discovery: what clusters exist, what hardware is free, who is using it and whether prerequisites are met."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..cluster_info import cluster_readiness, fetch_overview, summarise_overview
from ..context import ToolContext
from ..errors import ToolError
from ..lens_api import LensApiError
from ..models_cache import list_volumes, volume_view
from ..registry import tool, upstream_error

_ACTIVE_DEPLOYMENT = {"deploying", "ready", "rolling_back"}
_LIVE_LEDGER_STATES = {"creating", "active", "deleting"}


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


Section = Literal["capacity", "nodes", "readiness", "usage", "storage", "gateway"]


class GetClusterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(description="Cluster id from list_clusters.")
    include: list[Section] = Field(
        default_factory=lambda: ["capacity", "readiness", "usage", "storage", "gateway"],
        description="Sections to return; each one costs extra Lens calls, so ask only for what you need. "
        "Default: capacity, readiness, usage, storage, gateway. Add 'nodes' for the per-node accelerator split.",
    )


def _my_deployments(ctx: ToolContext, cluster_id: str) -> list[Any]:
    return [
        row
        for row in ctx.ledger.list_resources(owner=ctx.owner, kind="deployment", cluster_id=cluster_id)
        if row.state in _LIVE_LEDGER_STATES
    ]


async def _list_one(ctx: ToolContext, item: dict[str, Any]) -> dict[str, Any]:
    cluster_id = item["id"]
    entry: dict[str, Any] = {
        "cluster_id": cluster_id,
        "name": item.get("name"),
        "ready": item.get("ready"),
        "my_active_deployments": len(_my_deployments(ctx, cluster_id)),
    }
    try:
        summary = summarise_overview(cluster_id, await fetch_overview(ctx, cluster_id, timeout=20))
    except (LensApiError, TimeoutError):
        return {**entry, "reachable": False, "accelerators": [], "accelerators_free": None}
    return {
        **entry,
        "reachable": True,
        "accelerators": [
            {key: value for key, value in accelerator.items() if key != "max_per_node"}
            for accelerator in summary["accelerators"]
        ],
        "accelerators_free": summary["accelerators_free"],
        "accelerators_total": sum(accelerator["total"] or 0 for accelerator in summary["accelerators"]),
    }


@tool(
    "list_clusters",
    "[R] List the Kubernetes clusters Lens manages with their accelerator family and model, free and total "
    "accelerators, and how many deployments you have running on each. Also reports which tool profiles this "
    "connection has. Start here to get a cluster_id.\n"
    "Side effects: none; resources: none; duration: synchronous, about a second per cluster; reversible: n/a.",
    _NoArgs,
    read_only=True,
)
async def list_clusters(ctx: ToolContext, args: _NoArgs) -> dict[str, Any]:
    data = await ctx.api.request("GET", "/cluster/clusters")
    clusters = await asyncio.gather(*(_list_one(ctx, item) for item in data.get("items") or []))
    return {"clusters": list(clusters), "profiles": sorted(ctx.scopes)}


async def _usage(ctx: ToolContext, cluster_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    executions = await ctx.api.request(
        "GET", "/deployments/executions", params={"cluster_id": cluster_id, "limit": 500}, timeout=30
    )
    mine = {row.lens_execution_id: row for row in _my_deployments(ctx, cluster_id) if row.lens_execution_id}
    owned: list[dict[str, Any]] = []
    others = 0
    others_accelerators = 0
    for item in executions.get("items") or []:
        if item.get("status") not in _ACTIVE_DEPLOYMENT:
            continue
        accelerators = (item.get("replicas") or 1) * (item.get("tensor_parallel_size") or 1)
        row = mine.get(item.get("execution_id"))
        if row:
            owned.append(
                {
                    "deployment_id": row.id,
                    "name": item.get("name"),
                    "model": item.get("model"),
                    "status": item.get("status"),
                    "accelerators_estimate": accelerators,
                }
            )
        else:
            others += 1
            others_accelerators += accelerators
    owned_accelerators = sum(item["accelerators_estimate"] for item in owned)
    allocated = summary["accelerators_allocated"]
    return {
        "mine": {"deployments": owned, "accelerators_estimate": owned_accelerators},
        "others": {
            "deployments": others,
            "accelerators_estimate": others_accelerators,
            "note": "Lens records no owner, so deployments you did not create here are not attributed to a person.",
        },
        "allocated_total": allocated,
        "unattributed_estimate": None
        if allocated is None
        else max(allocated - owned_accelerators - others_accelerators, 0),
        "note": "Accelerators per deployment are estimated as replicas x tensor_parallel_size.",
    }


def _gateway_section(gateway: dict[str, Any] | None) -> dict[str, Any]:
    if gateway is None:
        return {"state": "unknown"}
    section: dict[str, Any] = {
        "state": gateway.get("gatewayState"),
        "gateway_ready": gateway.get("gatewayReady"),
        "epp_ready": gateway.get("ippReady"),
        "members": len(gateway.get("members") or []),
    }
    message = (gateway.get("lastOperation") or {}).get("message")
    if message:
        section["untrusted_text"] = {"last_operation": str(message)[:300]}
    return section


@tool(
    "get_cluster",
    "[R] Show one cluster: accelerator capacity (family, model, memory per device, free/total), usage split into "
    "your deployments and others', prerequisite readiness (nodes, Gateway API and inference-extension CRDs, device "
    "plugin, LeaderWorkerSet, HF token secret), model-cache storage volumes and gateway state. Reads Lens status "
    "only and never probes the cluster. Use include to limit the sections. Guides that fit the cluster: "
    "list_guides(cluster_id).\n"
    "Side effects: none; resources: none; duration: synchronous, a few seconds; reversible: n/a.",
    GetClusterArgs,
    read_only=True,
)
async def get_cluster(ctx: ToolContext, args: GetClusterArgs) -> dict[str, Any]:
    include = set(args.include)
    try:
        overview = await fetch_overview(ctx, args.cluster_id)
    except LensApiError as error:
        if error.status == 404:
            raise ToolError(
                "not_found", f"No cluster {args.cluster_id}.", field="cluster_id", hint="Call list_clusters."
            ) from error
        raise upstream_error(error) from error
    summary = summarise_overview(args.cluster_id, overview)
    result: dict[str, Any] = {"cluster_id": args.cluster_id, "name": summary["name"]}
    if "capacity" in include:
        result["capacity"] = {
            "accelerators": summary["accelerators"],
            "free": summary["accelerators_free"],
            "allocated": summary["accelerators_allocated"],
            "limit": summary["accelerator_limit"],
        }
    if "nodes" in include:
        result["nodes"] = summary["nodes"]
    sections: dict[str, Any] = {}
    if include & {"readiness", "gateway"}:
        sections["readiness"] = cluster_readiness(ctx, args.cluster_id, overview)
    if "usage" in include:
        sections["usage"] = _usage(ctx, args.cluster_id, summary)
    if "storage" in include:
        sections["storage"] = list_volumes(ctx, args.cluster_id)
    gathered = await asyncio.gather(*sections.values(), return_exceptions=True)
    outcome = dict(zip(sections, gathered, strict=True))
    for name, value in outcome.items():
        if isinstance(value, BaseException):
            if not isinstance(value, LensApiError):
                raise value
            result.setdefault("unavailable", {})[name] = value.message()
    readiness = outcome.get("readiness")
    if isinstance(readiness, dict):
        if "readiness" in include:
            result["readiness"] = {"overall": readiness["overall"], "items": readiness["items"]}
        if "gateway" in include:
            result["gateway"] = _gateway_section(readiness["gateway"])
    if isinstance(outcome.get("usage"), dict):
        result["usage"] = outcome["usage"]
    if isinstance(outcome.get("storage"), list):
        result["storage"] = [volume_view(ctx, volume) for volume in outcome["storage"]]
    return result
