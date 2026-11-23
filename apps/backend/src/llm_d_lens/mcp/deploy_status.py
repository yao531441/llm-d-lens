"""Live deployment status: Lens execution, run case and pods folded into one agent-facing view."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .context import ToolContext
from .deployment_refs import DeploymentRef, get_run, latest_case
from .lens_api import LensApiError

ROLE_LABEL = "llm-d.ai/role"
ROUTER_LABEL = "llm-d-router-gateway"

# Lens execution / case statuses -> the statuses agents see.
STATUS_MAP = {
    "draft": "pending",
    "validated": "pending",
    "rendered": "pending",
    "queued": "pending",
    "rendering": "pending",
    "deploying": "starting",
    "ready": "ready",
    "failed": "failed",
    "rolling_back": "stopping",
    "cancelling": "stopping",
    "rolled_back": "stopped",
    "stopped": "stopped",
    "cancelled": "stopped",
    "cleaned": "deleted",
    "cleaned_up": "deleted",
}
SETTLED = {"ready", "failed", "stopped", "deleted"}


def public_status(lens_status: str | None) -> str:
    return STATUS_MAP.get((lens_status or "").lower(), "pending")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def failure_text(failure: Any) -> str | None:
    if not failure:
        return None
    if isinstance(failure, dict):
        failure = failure.get("message") or failure.get("detail") or failure.get("reason") or json.dumps(failure)
    return str(failure)[:1000]


async def fetch_execution(ctx: ToolContext, execution_id: str) -> dict[str, Any] | None:
    try:
        return await ctx.api.request("GET", f"/deployments/executions/{execution_id}")
    except LensApiError as error:
        if error.status == 404:
            return None
        raise


def pod_role(labels: dict[str, str]) -> str | None:
    return labels.get(ROLE_LABEL) or ("router" if ROUTER_LABEL in labels else None)


async def fetch_pods(ctx: ToolContext, execution_id: str) -> list[dict[str, Any]]:
    """Pods that belong to the deployment; the namespace may also hold harness or tooling pods."""
    data = await ctx.api.request("GET", f"/deployments/executions/{execution_id}/pods", timeout=30)
    pods = []
    for pod in data.get("pods") or []:
        role = pod_role(pod.get("labels") or {})
        if role is None:
            continue
        pods.append(
            {
                "name": pod.get("name"),
                "role": role,
                "phase": pod.get("phase"),
                "ready": bool(pod.get("ready")),
                "restarts": pod.get("restarts") or 0,
                "reason": pod.get("status_reason"),
                "node": pod.get("node_name"),
                "containers": [
                    {
                        "name": c.get("name"),
                        "ready": bool(c.get("ready")),
                        "state": c.get("state"),
                        "restarts": c.get("restart_count") or 0,
                    }
                    for c in pod.get("containers") or []
                ],
            }
        )
    return pods


def default_container(pod: dict[str, Any]) -> str | None:
    """The model-server (or router) container; sidecars come second."""
    names = [c["name"] for c in pod["containers"]]
    return next(
        (name for name in ("vllm", "modelserver", "sglang", "epp") if name in names), names[0] if names else None
    )


def desired_replicas(execution: dict[str, Any], role: str) -> int | None:
    """Replicas Lens reports for the serving (decode) role; it reports nothing for prefill or the router."""
    return execution.get("replicas") if role in ("serving", "decode") else None


def role_readiness(pods: list[dict[str, Any]], execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for pod in pods:
        entry = roles.setdefault(
            pod["role"], {"ready": 0, "total": 0, "desired": desired_replicas(execution, pod["role"])}
        )
        entry["total"] += 1
        entry["ready"] += int(pod["ready"])
    return roles


async def load_view(ctx: ToolContext, ref: DeploymentRef) -> dict[str, Any]:
    """Status, execution payload and pods for one deployment."""
    view: dict[str, Any] = {"execution": ref.execution or None, "pods": [], "failure": None, "case": None}
    if ref.execution:
        view["status"] = public_status(ref.execution.get("status"))
        view["failure"] = failure_text(ref.execution.get("failure"))
        if ref.execution.get("namespace") and view["status"] != "deleted":
            view["pods"] = await fetch_pods(ctx, ref.execution_id)
        return view
    case = latest_case(await get_run(ctx, ref.run_id) or {}) if ref.run_id else None
    view["case"] = case
    if case:
        view["status"] = public_status(case.get("status"))
        view["failure"] = failure_text(case.get("failure"))
    else:
        view["status"] = "pending"
    return view
