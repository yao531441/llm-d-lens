"""Plan and install shared cluster components through the existing Lens installers (never accelerator drivers)."""

from __future__ import annotations

from typing import Any

from .context import ToolContext
from .errors import ToolError

LOG_TAIL = 20


def _checks(preflight: dict[str, Any]) -> dict[str, Any]:
    checks = preflight.get("checks") or []
    failed = [item for item in checks if not item.get("passed")]
    output: dict[str, Any] = {
        "allowed": bool(preflight.get("allowed")),
        "checks": [
            {"name": item.get("name"), "passed": bool(item.get("passed")), "blocking": bool(item.get("blocking"))}
            for item in checks
        ],
        "warnings": len(preflight.get("warnings") or []),
    }
    texts = [f"{item.get('name')}: {item.get('message')}" for item in failed] + [
        str(item) for item in preflight.get("warnings") or []
    ]
    if texts:
        output["untrusted_text"] = texts[:20]
    return output


async def _accelerator_defaults(ctx: ToolContext, accelerator: str) -> dict[str, Any]:
    catalog = await ctx.api.request("GET", "/monitoring/accelerators")
    entry = next((item for item in catalog.get("accelerators") or [] if item.get("type") == accelerator), None)
    if entry is None:
        raise ToolError("not_found", f"Unknown accelerator '{accelerator}'.", field="accelerator")
    return entry


async def _accelerator_body(ctx: ToolContext, accelerator: str, access_mode: str) -> dict[str, Any]:
    entry = await _accelerator_defaults(ctx, accelerator)
    if access_mode not in (entry.get("access_modes") or []):
        raise ToolError(
            "unsupported_combination",
            f"{accelerator} supports access modes {entry.get('access_modes')}, not '{access_mode}'.",
            field="access_mode",
        )
    return {"accelerator": accelerator, "access_mode": access_mode, "namespace": entry["default_namespace"]}


def _operation(operation: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "operation_id": operation.get("operation_id"),
        "operation_status": operation.get("status"),
        "phase": operation.get("phase"),
    }
    texts = [str(operation["error"])[:500]] if operation.get("error") else []
    texts.extend(str(line)[:300] for line in (operation.get("logs") or [])[-LOG_TAIL:])
    if texts:
        output["untrusted_text"] = texts
    return output


def prepull_state(status: dict[str, Any]) -> dict[str, Any]:
    nodes = status.get("nodes") or []
    return {
        "state": status.get("state"),
        "desired_nodes": status.get("desired"),
        "ready_nodes": sum(1 for node in nodes if node.get("ready")),
        "not_ready_nodes": [node.get("node") for node in nodes if not node.get("ready")],
        "images": status.get("images") or [],
    }


async def plan_component(
    ctx: ToolContext, cluster_id: str, component: str, accelerator: str, access_mode: str
) -> dict[str, Any]:
    """Read-only state and preflight of one component; changes nothing on the cluster."""
    base = f"/cluster/clusters/{cluster_id}"
    params = {"cluster_id": cluster_id}
    if component == "gateway-api-crds":
        bundles = await ctx.api.request("GET", f"{base}/crds")
        return {"bundles": bundles, "installed": all(state == "ready" for state in bundles.values())}
    if component == "lws":
        lws = await ctx.api.request("GET", f"{base}/lws")
        output = {key: lws.get(key) for key in ("version", "namespace", "crd", "controller")}
        return {**output, "installed": bool(lws.get("ready"))}
    if component == "software-downloads":
        status = await ctx.api.request("GET", f"{base}/software-downloads")
        states = {key: (value or {}).get("state") for key, value in status.items()}
        return {"states": states, "installed": all(state == "ready" for state in states.values())}
    if component == "monitoring-stack":
        preflight = await ctx.api.request(
            "POST", "/monitoring/cluster-stack/installations/preflight", params=params, body={}
        )
        status = await ctx.api.request("GET", "/monitoring/cluster-stack/status", params=params)
        return {
            **_checks(preflight),
            "installed": status.get("status") in {"ready", "degraded"},
            "current_status": status.get("status"),
            "namespace": status.get("namespace"),
        }
    if component == "accelerator-monitoring":
        body = await _accelerator_body(ctx, accelerator, access_mode)
        preflight = await ctx.api.request(
            "POST", f"/monitoring/accelerators/{accelerator}/installations/preflight", params=params, body=body
        )
        status = await ctx.api.request("GET", f"/monitoring/accelerators/{accelerator}/status", params=params)
        return {
            **_checks(preflight),
            "installed": status.get("status") == "ready",
            "current_status": status.get("status"),
            "namespace": body["namespace"],
        }
    if component == "image-prepull":
        return prepull_state(await ctx.api.request("GET", f"{base}/images/prepull"))
    operation = await ctx.api.request("POST", "/model-service/admin/gateway/preflight", body={"clusterId": cluster_id})
    gateway = await ctx.api.request("GET", "/model-service/admin/gateway/status")
    state = next((item for item in gateway.get("clusters") or [] if item.get("clusterId") == cluster_id), {})
    return {
        "preflight_ok": operation.get("status") == "succeeded",
        "installed": state.get("gatewayState") == "installed",
        "gateway_namespace": state.get("gatewayNamespace"),
        "untrusted_text": [str(operation.get("message") or "")[:500]],
    }


async def operation_progress(ctx: ToolContext, component: str, accelerator: str, operation_id: str) -> dict[str, Any]:
    if component == "monitoring-stack":
        path = f"/monitoring/cluster-stack/operations/{operation_id}"
    elif component == "accelerator-monitoring":
        path = f"/monitoring/accelerators/{accelerator}/operations/{operation_id}"
    else:
        raise ToolError(
            "invalid_argument",
            "operation_id is only supported for monitoring-stack and accelerator-monitoring.",
            field="operation_id",
        )
    return _operation(await ctx.api.request("GET", path))


async def install_component(
    ctx: ToolContext, cluster_id: str, component: str, accelerator: str, access_mode: str, reinstall: bool
) -> dict[str, Any]:
    base = f"/cluster/clusters/{cluster_id}"
    params = {"cluster_id": cluster_id}
    if component == "gateway-api-crds":
        bundles = await ctx.api.request("POST", f"{base}/crds", timeout=600)
        failed = [key for key, value in bundles.items() if not value.get("ok")]
        return {
            "status": "failed" if failed else "installed",
            "bundles": {key: bool(value.get("ok")) for key, value in bundles.items()},
            "untrusted_text": [f"{key}: {bundles[key].get('message')}" for key in failed],
        }
    if component == "lws":
        lws = await ctx.api.request("POST", f"{base}/lws", timeout=900)
        output: dict[str, Any] = {"status": "installed" if lws.get("ready") else "failed"}
        if lws.get("message"):
            output["untrusted_text"] = [str(lws["message"])[:500]]
        return output
    if component == "software-downloads":
        status = await ctx.api.request("POST", f"{base}/software-downloads", timeout=120)
        return {"status": "started", "states": {key: (value or {}).get("state") for key, value in status.items()}}
    if component == "monitoring-stack":
        operation = await ctx.api.request(
            "POST", "/monitoring/cluster-stack/installations", params=params, body={"reinstall": reinstall}, timeout=120
        )
        return {"status": "started", **_operation(operation)}
    if component == "accelerator-monitoring":
        body = await _accelerator_body(ctx, accelerator, access_mode)
        operation = await ctx.api.request(
            "POST", f"/monitoring/accelerators/{accelerator}/installations", params=params, body=body, timeout=120
        )
        return {"status": "started", **_operation(operation)}
    operation = await ctx.api.request(
        "POST", "/model-service/admin/gateway/install", body={"clusterId": cluster_id}, timeout=900
    )
    return {
        "status": "installed" if operation.get("status") == "succeeded" else operation.get("status"),
        "operation_id": operation.get("id"),
        "untrusted_text": [str(operation.get("message") or "")[:500]],
    }
