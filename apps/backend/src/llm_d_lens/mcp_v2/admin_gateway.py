"""Operate the shared model gateway: data-plane replicas, IPP config and route reconciliation."""

from __future__ import annotations

from typing import Any

import yaml

from .context import ToolContext
from .errors import ToolError

IPP_KIND = "PayloadProcessorConfig"


def _failed(operation: dict[str, Any], what: str) -> ToolError:
    return ToolError(
        "operation_failed",
        f"Lens reported that {what} failed.",
        hint="Read plan_cluster_component(component='model-gateway') for the current state before retrying.",
        untrusted_text=[str(operation.get("message") or "")[:500]],
    )


def validate_ipp_config(config: str) -> None:
    try:
        document = yaml.safe_load(config)
    except yaml.YAMLError as error:
        raise ToolError("invalid_argument", "ipp_config is not valid YAML.", field="ipp_config") from error
    if not isinstance(document, dict) or document.get("kind") != IPP_KIND:
        raise ToolError("invalid_argument", f"ipp_config must be a {IPP_KIND} document.", field="ipp_config")


async def _gateway_target(ctx: ToolContext, cluster_id: str) -> tuple[str, str]:
    """The data-plane Deployment Lens installed for this cluster; never a name supplied by the caller."""
    status = await ctx.api.request("GET", "/model-service/admin/gateway/status")
    cluster = next((item for item in status.get("clusters") or [] if item.get("clusterId") == cluster_id), None)
    if not cluster or cluster.get("gatewayState") != "installed":
        raise ToolError(
            "conflict_state",
            f"No Lens-installed gateway on cluster {cluster_id}.",
            field="cluster_id",
            hint="Install it with install_cluster_component(component='model-gateway').",
        )
    if cluster.get("gatewayProvider") != "istio":
        raise ToolError(
            "unsupported_combination",
            f"Replica changes are only supported for the istio provider, not '{cluster.get('gatewayProvider')}'.",
            field="replicas",
        )
    namespace, gateway_name = cluster.get("gatewayNamespace"), cluster.get("gatewayName")
    if not namespace or not gateway_name:
        raise ToolError(
            "conflict_state",
            f"Lens did not report the gateway namespace and name for cluster {cluster_id}; nothing was changed.",
            field="cluster_id",
            hint="Run plan_cluster_component(component='model-gateway') and reconcile or reinstall the gateway.",
        )
    return namespace, f"{gateway_name}-istio"


async def configure_gateway(
    ctx: ToolContext, cluster_id: str, replicas: int | None, ipp_config: str | None
) -> dict[str, Any]:
    if ipp_config is not None:
        validate_ipp_config(ipp_config)
    result: dict[str, Any] = {"cluster_id": cluster_id}
    if replicas is not None:
        namespace, name = await _gateway_target(ctx, cluster_id)
        operation = await ctx.api.request(
            "POST",
            "/model-service/admin/gateway/scale",
            body={"clusterId": cluster_id, "namespace": namespace, "name": name, "replicas": replicas},
            timeout=90,
        )
        if operation.get("status") != "succeeded":
            raise _failed(operation, "scaling the gateway")
        result["gateway_replicas"] = replicas
    if ipp_config is not None:
        operation = await ctx.api.request(
            "POST",
            "/model-service/admin/gateway/ipp-config",
            body={"clusterId": cluster_id, "config": ipp_config},
            timeout=300,
        )
        if operation.get("status") != "succeeded":
            raise _failed(operation, "applying the IPP config")
        result["ipp_config_applied"] = True
    return result


async def reconcile_gateway(ctx: ToolContext) -> dict[str, Any]:
    scheduled = await ctx.api.request("POST", "/model-service/admin/gateway/reconcile", timeout=60)
    return {"scheduled": bool(scheduled.get("scheduled")), "clusters": scheduled.get("clusters") or []}
