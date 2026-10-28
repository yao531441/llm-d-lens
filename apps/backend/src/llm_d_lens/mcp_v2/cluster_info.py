"""Shared cluster summarisation used by tools and the planner."""

from __future__ import annotations

from typing import Any

from .context import ToolContext


def accelerator_family(profile_id: str) -> str:
    """Map a Lens accelerator profile id (e.g. intel-xpu, nvidia) to the guide catalog id (xpu or gpu)."""
    return "xpu" if "xpu" in profile_id.lower() else "gpu"


async def cluster_summary(ctx: ToolContext, cluster_id: str) -> dict[str, Any]:
    overview = await ctx.api.request("GET", "/cluster/overview", params={"clusterId": cluster_id})
    kubernetes = overview.get("kubernetes") or {}
    hardware = kubernetes.get("hardware") or {}
    nodes = [
        {
            "name": node.get("name"),
            "role": node.get("role"),
            "ready": node.get("ready"),
            "schedulable": not node.get("schedulingDisabled"),
            "accelerators": node.get("gpuByProfile") or {},
            "accelerator_models": node.get("gpuModelsByProfile") or {},
        }
        for node in kubernetes.get("nodes") or []
    ]
    components = {
        name: value.get("status")
        for name, value in (kubernetes.get("components") or {}).items()
        if isinstance(value, dict)
    }
    return {
        "cluster_id": cluster_id,
        "accelerators": [
            {
                "profile": item.get("id"),
                "family": accelerator_family(item.get("id") or ""),
                "total": item.get("gpuCount"),
                "models": item.get("models") or [],
            }
            for item in hardware.get("accelerators") or []
        ],
        "accelerators_free": hardware.get("usableGpuCount"),
        "accelerators_allocated": hardware.get("allocatedGpuCount"),
        "nodes": nodes,
        "components": components,
    }
