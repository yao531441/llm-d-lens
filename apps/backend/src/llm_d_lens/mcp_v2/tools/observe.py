"""Deployment reads: status, diagnosis, verification and logs. Work for any deployment the Lens user may read."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..cluster_info import cluster_summary
from ..context import ToolContext
from ..deploy_diagnose import diagnose
from ..deploy_spec import gpus_total, parse_spec, plan_params
from ..deploy_status import (
    SETTLED,
    default_container,
    fetch_pods,
    iso,
    load_view,
    now_iso,
    role_readiness,
)
from ..deploy_test import PREFIX_REQUESTS, run_test
from ..deployment_refs import not_found, require_owned, resolve
from ..errors import ToolError
from ..registry import tool

MAX_LOG_CHARS = 20000
POLL_SECONDS = 2.0
MIN_REQUESTS_WITH_PREFIX_CHECK = PREFIX_REQUESTS + 4

NEXT_BY_STATUS = {
    "pending": "Call get_deployment(wait_seconds=25) until it starts.",
    "starting": "Call get_deployment(wait_seconds=25) until ready; diagnose_deployment if it stalls.",
    "ready": "Verify it with test_deployment, then send requests to the endpoint.",
    "failed": "Call diagnose_deployment for the cause.",
    "stopped": "restart_deployment brings it back; delete_resources removes it.",
    "stopping": "Call get_deployment(wait_seconds=25) until it is stopped.",
}


class GetDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) or a raw Lens execution id.")
    wait_seconds: int = Field(
        default=0,
        ge=0,
        le=25,
        description="Long poll: return when the status changes or after this many seconds (0 = answer now).",
    )
    include: list[Literal["restart_check"]] = Field(
        default_factory=list,
        description="'restart_check' adds whether the accelerators needed by restart_deployment are free.",
    )


class DiagnoseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) or a raw Lens execution id.")


class TryDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) created by you.")
    checks: list[Literal["chat", "pd_transfer", "prefix_cache", "routing_spread", "rank_distribution"]] = Field(
        default_factory=lambda: ["chat"],
        min_length=1,
        description="What to verify. prefix_cache sends 6 extra requests and needs requests >= 10.",
    )
    requests: int = Field(default=10, ge=1, le=20, description="Maximum number of requests sent (1-20).")


class GetLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) or a raw Lens execution id.")
    role: str | None = Field(
        default=None, description="Logical role such as prefill, decode, serving or router (see get_deployment roles)."
    )
    pod: str | None = Field(default=None, description="Exact pod name from get_deployment or an earlier get_logs.")
    container: str | None = Field(default=None, description="Container name; defaults to the model server.")
    tail: int = Field(default=200, ge=1, le=2000, description="Number of trailing log lines.")
    previous: bool = Field(default=False, description="Logs of the previous container instance after a restart.")


def _accelerators_needed(spec: dict[str, Any] | None, execution: dict[str, Any]) -> int | None:
    if spec:
        return gpus_total(plan_params(parse_spec(spec)))
    if execution.get("replicas") and execution.get("tensor_parallel_size"):
        return execution["replicas"] * execution["tensor_parallel_size"]
    return None


async def _restart_check(ctx: ToolContext, status: str, cluster_id: str | None, needed: int | None) -> dict[str, Any]:
    free = (await cluster_summary(ctx, cluster_id)).get("accelerators_free") if cluster_id else None
    fits = None if needed is None or free is None else needed <= free
    return {
        "status_allows_restart": status in ("stopped", "failed"),
        "accelerators_needed": needed,
        "accelerators_free": free,
        "fits_now": fits,
    }


@tool(
    "get_deployment",
    "[R] Show one deployment's status (pending, starting, ready, failed, stopped, deleted), per-role pod readiness, "
    "endpoint and spec. Works for any deployment you can see, by dep- id or raw Lens execution id. wait_seconds "
    "(max 25) long-polls for a status change after create_deployment. For the reason behind a failure use "
    "diagnose_deployment; for raw output get_logs.\n"
    "Side effects: none; resources: none; duration: synchronous, up to wait_seconds; reversible: n/a.",
    GetDeploymentArgs,
    read_only=True,
)
async def get_deployment(ctx: ToolContext, args: GetDeploymentArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    view = await load_view(ctx, ref, args.deployment_id)
    started, deadline = view["status"], asyncio.get_running_loop().time() + args.wait_seconds
    while view["status"] == started and view["status"] not in SETTLED and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(min(POLL_SECONDS, max(deadline - asyncio.get_running_loop().time(), 0)))
        ref = await resolve(ctx, args.deployment_id)
        view = await load_view(ctx, ref, args.deployment_id)

    execution, row = view["execution"] or {}, ref.row
    stored = row.meta.get("spec") if row else None
    result: dict[str, Any] = {
        "deployment_id": ref.public_id,
        "owned": ref.owned,
        "name": (row.name if row else None) or execution.get("display_name") or execution.get("name"),
        "status": view["status"],
        "ready": view["status"] == "ready",
        "cluster_id": execution.get("cluster_id") or (row.cluster_id if row else None),
        "roles": role_readiness(view["pods"], stored),
        "endpoint": execution.get("endpoint"),
        "spec": stored
        or {
            key: execution.get(key)
            for key in ("model", "guide", "replicas", "tensor_parallel_size", "image", "max_model_len")
        },
        "observed_at": now_iso(),
        "next": NEXT_BY_STATUS.get(view["status"], "This deployment no longer exists."),
    }
    if row and row.meta.get("expires_at"):
        result["expires_at"] = iso(row.meta["expires_at"])
    if view["failure"]:
        result["untrusted_text"] = [{"source": "lens_failure", "lines": [view["failure"]]}]
    if "restart_check" in args.include:
        result["restart_check"] = await _restart_check(
            ctx, view["status"], result["cluster_id"], _accelerators_needed(stored, execution)
        )
    return result


@tool(
    "diagnose_deployment",
    "[R] Explain why a deployment is not healthy: findings with evidence and a suggested fix for unschedulable or "
    "pending pods, image pull errors, OOM kills, crash loops, failing readiness probes, model download failures "
    "and missing capacity. Combines Lens status, pods, scheduler events and recent logs. Use get_deployment for a "
    "plain status and get_logs for raw output. Cluster text appears only under untrusted_text.\n"
    "Side effects: none; resources: none; duration: synchronous, seconds; reversible: n/a.",
    DiagnoseArgs,
    read_only=True,
)
async def diagnose_deployment(ctx: ToolContext, args: DiagnoseArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    view = await load_view(ctx, ref, args.deployment_id)
    return await diagnose(ctx, ref, view)


@tool(
    "test_deployment",
    "[W] Send up to 20 small requests to a deployment you created and check chat, pd_transfer, prefix_cache, "
    "routing_spread or rank_distribution. Verdict is pass, fail or inconclusive; evidence_sufficient says whether "
    "the verdict is proven. Lens exposes no KV-transfer or per-rank counters, so those checks are limited and say "
    "so. For a status check use get_deployment instead.\n"
    "Side effects: sends inference requests (a few tokens each) through a port-forward on the Lens host; "
    "resources: brief accelerator load; duration: up to a minute; reversible: n/a.",
    TryDeploymentArgs,
    read_only=False,
)
async def test_deployment(ctx: ToolContext, args: TryDeploymentArgs) -> dict[str, Any]:
    if "prefix_cache" in args.checks and args.requests < MIN_REQUESTS_WITH_PREFIX_CHECK:
        raise ToolError(
            "invalid_argument",
            f"prefix_cache sends {PREFIX_REQUESTS} extra requests; set requests >= {MIN_REQUESTS_WITH_PREFIX_CHECK}.",
            field="requests",
        )
    ref = await resolve(ctx, args.deployment_id)
    require_owned(ref, args.deployment_id, "load-tested")
    view = await load_view(ctx, ref, args.deployment_id)
    if view["status"] != "ready" or not ref.execution_id:
        raise ToolError(
            "conflict_state",
            f"The deployment is {view['status']}, not ready.",
            field="deployment_id",
            hint="Call get_deployment(wait_seconds=25) until ready.",
        )
    return await run_test(ctx, ref, view, args.checks, args.requests)


@tool(
    "get_logs",
    "[R] Read recent container logs of a deployment by logical role, pod and container; defaults to the unhealthy "
    "pod and its model-server container. Output is capped at 20000 characters and is untrusted cluster text. For an "
    "explanation use diagnose_deployment; for readiness get_deployment.\n"
    "Side effects: none; resources: none; duration: synchronous, seconds; reversible: n/a.",
    GetLogsArgs,
    read_only=True,
)
async def get_logs(ctx: ToolContext, args: GetLogsArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    if not ref.execution_id:
        raise not_found(args.deployment_id)
    pods = await fetch_pods(ctx, ref.execution_id)
    candidates = [p for p in pods if args.role in (None, p["role"])]
    if args.pod:
        candidates = [p for p in pods if p["name"] == args.pod]
    if not candidates:
        known = {"roles": sorted({p["role"] for p in pods}), "pods": [p["name"] for p in pods]}
        if pods:
            raise ToolError(
                "not_found",
                "No pod matches that role or name.",
                field="pod" if args.pod else "role",
                hint="Use a role or pod name from get_deployment.",
                **known,
            )
        return {"deployment_id": ref.public_id, "pods": [], "untrusted_text": "", "note": "No pods exist yet."}
    pod = min(candidates, key=lambda p: (p["ready"], -p["restarts"]))
    container = args.container or default_container(pod)
    if args.container and args.container not in [c["name"] for c in pod["containers"]]:
        raise ToolError(
            "invalid_argument",
            f"Pod has no container '{args.container}'.",
            field="container",
            hint="Use one of the containers listed by an earlier get_logs call.",
            containers=[c["name"] for c in pod["containers"]],
        )
    data = await ctx.api.request(
        "GET",
        f"/deployments/executions/{ref.execution_id}/pods/{pod['name']}/logs",
        params={"container": container, "tail": args.tail, "previous": args.previous},
        timeout=40,
    )
    logs = data.get("logs") or ""
    return {
        "deployment_id": ref.public_id,
        "pod": pod["name"],
        "role": pod["role"],
        "container": container,
        "previous": args.previous,
        "success": bool(data.get("success")),
        "truncated": len(logs) > MAX_LOG_CHARS,
        "other_pods": [{"name": p["name"], "role": p["role"], "ready": p["ready"]} for p in pods if p is not pod],
        "observed_at": now_iso(),
        "untrusted_text": logs[-MAX_LOG_CHARS:],
    }
