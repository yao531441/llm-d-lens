"""Cluster discovery: what clusters exist and what hardware is free."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..cluster_info import cluster_summary
from ..context import ToolContext
from ..registry import tool


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetClusterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(description="Cluster id from list_clusters.")


@tool(
    "list_clusters",
    "List the Kubernetes clusters Lens manages and whether each is ready. Start here to get a cluster_id.",
    _NoArgs,
    read_only=True,
)
async def list_clusters(ctx: ToolContext, args: _NoArgs) -> dict[str, Any]:
    data = await ctx.api.request("GET", "/cluster/clusters")
    return {
        "clusters": [
            {"cluster_id": item["id"], "name": item.get("name"), "ready": item.get("ready")}
            for item in data.get("items") or []
        ]
    }


@tool(
    "get_cluster",
    "Show one cluster's accelerators (type, total, free), nodes and platform components. Use it to size a "
    "deployment before planning.",
    GetClusterArgs,
    read_only=True,
)
async def get_cluster(ctx: ToolContext, args: GetClusterArgs) -> dict[str, Any]:
    return await cluster_summary(ctx, args.cluster_id)
