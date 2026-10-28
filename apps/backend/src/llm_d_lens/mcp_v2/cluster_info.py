"""Shared cluster summarisation used by tools and the planner."""

from __future__ import annotations

import asyncio
from typing import Any

from .context import ToolContext
from .lens_api import LensApiError

BYTES_PER_GIB = 1024**3


def accelerator_family(profile_id: str) -> str:
    """Map a Lens accelerator profile id (e.g. intel-xpu, nvidia) to the guide catalog id (xpu or gpu)."""
    return "xpu" if "xpu" in profile_id.lower() else "gpu"


def _device_memory(nodes: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
    """Per accelerator profile: memory GiB of one device and the largest device count on a single node.

    Only nodes with a single accelerator profile are used, because Lens reports VRAM per device without a profile.
    """
    result: dict[str, tuple[float, int]] = {}
    for node in nodes:
        by_profile = node.get("gpuByProfile") or {}
        for profile, count in by_profile.items():
            sizes = [gpu.get("vramTotalBytes") or 0 for gpu in node.get("gpus") or []]
            if not sizes and node.get("vramBytes") and count:
                sizes = [node["vramBytes"] / count]
            memory = round(max(sizes) / BYTES_PER_GIB, 1) if sizes and len(by_profile) == 1 else 0.0
            known_memory, known_count = result.get(profile, (0.0, 0))
            result[profile] = (known_memory or memory, max(known_count, int(count or 0)))
    return result


def summarise_overview(cluster_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    kubernetes = overview.get("kubernetes") or {}
    hardware = kubernetes.get("hardware") or {}
    raw_nodes = kubernetes.get("nodes") or []
    nodes = [
        {
            "name": node.get("name"),
            "role": node.get("role"),
            "ready": node.get("ready"),
            "schedulable": not node.get("schedulingDisabled"),
            "accelerators": node.get("gpuByProfile") or {},
            "accelerator_models": node.get("gpuModelsByProfile") or {},
        }
        for node in raw_nodes
    ]
    components = {
        name: value.get("status")
        for name, value in (kubernetes.get("components") or {}).items()
        if isinstance(value, dict)
    }
    memory = _device_memory(raw_nodes)
    return {
        "cluster_id": cluster_id,
        "name": (overview.get("cluster") or {}).get("name"),
        "accelerators": [
            {
                "profile": item.get("id"),
                "family": accelerator_family(item.get("id") or ""),
                "total": item.get("gpuCount"),
                "models": item.get("models") or [],
                "memory_gib_per_device": memory.get(item.get("id"), (0.0, 0))[0] or None,
                "max_per_node": memory.get(item.get("id"), (0.0, 0))[1] or None,
            }
            for item in hardware.get("accelerators") or []
        ],
        "accelerators_free": hardware.get("usableGpuCount"),
        "accelerators_allocated": hardware.get("allocatedGpuCount"),
        "accelerator_limit": hardware.get("configuredGpuLimit"),
        "nodes": nodes,
        "components": components,
    }


async def fetch_overview(ctx: ToolContext, cluster_id: str, timeout: float = 60) -> dict[str, Any]:
    return await ctx.api.request("GET", "/cluster/overview", params={"clusterId": cluster_id}, timeout=timeout)


async def cluster_summary(ctx: ToolContext, cluster_id: str) -> dict[str, Any]:
    return summarise_overview(cluster_id, await fetch_overview(ctx, cluster_id))


async def read_or_none(ctx: ToolContext, path: str, params: dict[str, Any] | None = None) -> Any:
    """GET that turns a Lens failure into None so one unreadable part does not fail a whole summary."""
    try:
        return await ctx.api.request("GET", path, params=params, timeout=30)
    except (LensApiError, TimeoutError):
        return None


_COMPONENT_STATUS = {"ready": "ready", "missing": "missing", "degraded": "degraded", "progressing": "degraded"}


def _item(component: str, status: str, *, required: bool, **extra: Any) -> dict[str, Any]:
    return {"component": component, "status": status, "required": required, **extra}


def _overall(items: list[dict[str, Any]]) -> str:
    required = [item["status"] for item in items if item["required"]]
    if not required or all(status == "unknown" for status in required):
        return "unknown"
    if all(status == "ready" for status in required):
        return "ready"
    return "partial" if "ready" in required else "not-ready"


def _gateway_entry(payload: Any, cluster_id: str) -> dict[str, Any] | None:
    for entry in (payload or {}).get("clusters") or []:
        if entry.get("clusterId") == cluster_id:
            return entry
    return None


def _gateway_status(gateway: dict[str, Any] | None) -> str:
    if gateway is None:
        return "unknown"
    if gateway.get("gatewayState") != "installed":
        return "missing"
    return "ready" if gateway.get("gatewayReady") and gateway.get("ippReady") else "degraded"


async def cluster_readiness(ctx: ToolContext, cluster_id: str, overview: dict[str, Any]) -> dict[str, Any]:
    """Prerequisite components for llm-d guides, from read-only Lens status endpoints (nothing is probed)."""
    crds, lws, version, gateways = await asyncio.gather(
        read_or_none(ctx, f"/cluster/clusters/{cluster_id}/crds"),
        read_or_none(ctx, f"/cluster/clusters/{cluster_id}/lws"),
        read_or_none(ctx, f"/cluster/clusters/{cluster_id}/kubernetes-version"),
        read_or_none(ctx, "/model-service/admin/gateway/status", {"clusterId": cluster_id}),
    )
    kubernetes = overview.get("kubernetes") or {}
    components = kubernetes.get("components") or {}
    nodes = kubernetes.get("nodes") or []
    ready_nodes = sum(1 for node in nodes if node.get("ready"))
    items = [
        _item(
            "nodes",
            "unknown" if not nodes else "ready" if ready_nodes == len(nodes) else "degraded",
            required=True,
            ready=ready_nodes,
            total=len(nodes),
        ),
        _item(
            "gateway-api-crds",
            "unknown" if crds is None else "ready" if crds.get("gateway_api") == "ready" else "missing",
            required=True,
        ),
        _item(
            "inference-extension-crds",
            "unknown"
            if crds is None
            else "ready"
            if crds.get("gateway_api_inference_extension") == "ready"
            else "missing",
            required=True,
        ),
        _item(
            "lws",
            "unknown" if lws is None else "ready" if lws.get("ready") else "missing",
            required=False,
            required_by="guides that use LeaderWorkerSet (for example wide expert parallelism)",
        ),
        _item(
            "hf-token-secret",
            "ready" if (overview.get("cluster") or {}).get("hfTokenSecret") else "missing",
            required=False,
            note="Needed for gated models; model downloads use the cluster's saved token.",
        ),
    ]
    if version is not None:
        items.append(
            _item(
                "kubernetes-version",
                "ready" if version.get("supported") else "degraded",
                required=False,
                version=version.get("version"),
                minimum=version.get("min_k8s_version"),
            )
        )
    for profile, value in (components.get("acceleratorDevicePlugins") or {}).items():
        if isinstance(value, dict) and (value.get("installed") or value.get("desired")):
            items.append(
                _item(
                    f"device-plugin:{profile}",
                    _COMPONENT_STATUS.get(value.get("status"), "unknown"),
                    required=True,
                    ready=value.get("ready"),
                    desired=value.get("desired"),
                )
            )
    monitoring = components.get("monitoring")
    if isinstance(monitoring, dict):
        items.append(_item("monitoring", _COMPONENT_STATUS.get(monitoring.get("status"), "unknown"), required=False))
    gateway = _gateway_entry(gateways, cluster_id)
    items.append(_item("model-gateway", _gateway_status(gateway), required=False))
    return {"overall": _overall(items), "items": items, "gateway": gateway}
