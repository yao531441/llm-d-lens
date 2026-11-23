"""Deployment reads: status, diagnosis, verification and logs. Work for any deployment the Lens user may read."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..deploy_diagnose import diagnose
from ..deploy_status import (
    SETTLED,
    default_container,
    fetch_pods,
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
    "pending": "Call deployment_get(wait_seconds=25) until it starts.",
    "starting": "Call deployment_get(wait_seconds=25) until ready; deployment_diagnose if it stalls.",
    "ready": "Verify it with deployment_test, then send requests to the endpoint.",
    "failed": "Call deployment_diagnose for the cause.",
    "stopped": "deployment_create(from_deployment_id=...) makes a fresh copy; resource_delete removes it.",
    "stopping": "Call deployment_get(wait_seconds=25) until it is stopped.",
}


class GetDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(
        description="Deployment id (a Lens run id; an execution id is also accepted) from deployment_create "
        "or resource_list."
    )
    wait_seconds: int = Field(
        default=0,
        ge=0,
        le=25,
        description="Long poll: return when the status changes or after this many seconds (0 = answer now).",
    )
    include: list[Literal["verifications"]] = Field(
        default_factory=list,
        description="'verifications' reads saved checks without inference.",
    )


class DiagnoseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(
        description="Deployment id (a Lens run id; an execution id is also accepted) from deployment_create "
        "or resource_list."
    )


class TryDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (a Lens run id; an execution id is also accepted).")
    checks: list[Literal["chat", "prefix_cache"]] = Field(
        default_factory=lambda: ["chat"],
        min_length=1,
        description="What to verify. prefix_cache uses 6 requests from the total budget and needs requests >= 10.",
    )
    requests: int = Field(default=10, ge=1, le=20, description="Maximum number of requests sent (1-20).")
    verification_id: str | None = Field(
        default=None, description="Read this existing verification instead of starting another."
    )
    wait_seconds: int = Field(default=25, ge=0, le=25, description="Wait up to this long for a persisted result.")


class GetLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(
        description="Deployment id (a Lens run id; an execution id is also accepted) from deployment_create "
        "or resource_list."
    )
    role: str | None = Field(
        default=None, description="Logical role such as prefill, decode, serving or router (see deployment_get roles)."
    )
    pod: str | None = Field(
        default=None, description="Exact pod name from deployment_get or an earlier deployment_logs."
    )
    container: str | None = Field(default=None, description="Container name; defaults to the model server.")
    tail: int = Field(default=200, ge=1, le=2000, description="Number of trailing log lines.")
    previous: bool = Field(default=False, description="Logs of the previous container instance after a restart.")


@tool(
    "deployment_get",
    "[R] Show one deployment's status (pending, starting, ready, failed, stopped, deleted), per-role pod readiness, "
    "endpoint and spec. Works for any deployment you can see, by Lens run id or execution id. wait_seconds "
    "(max 25) long-polls for a status change after deployment_create. For the reason behind a failure use "
    "deployment_diagnose; for raw output deployment_logs.\n"
    "Side effects: none; resources: none; duration: synchronous, up to wait_seconds; reversible: n/a.",
    GetDeploymentArgs,
    read_only=True,
)
async def get_deployment(ctx: ToolContext, args: GetDeploymentArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    view = await load_view(ctx, ref)
    started, deadline = view["status"], asyncio.get_running_loop().time() + args.wait_seconds
    while view["status"] == started and view["status"] not in SETTLED and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(min(POLL_SECONDS, max(deadline - asyncio.get_running_loop().time(), 0)))
        ref = await resolve(ctx, args.deployment_id)
        view = await load_view(ctx, ref)

    execution = view["execution"] or {}
    result: dict[str, Any] = {
        "deployment_id": ref.public_id,
        "owned": ref.owned,
        "name": execution.get("display_name") or execution.get("name"),
        "run_id": ref.run_id,
        "case_id": ref.case_id,
        "execution_id": ref.execution_id,
        "ownership": execution.get("ownership", "unknown"),
        "capabilities": execution.get("capabilities") or {},
        "description": execution.get("description"),
        "namespace": execution.get("namespace"),
        "status": view["status"],
        "ready": view["status"] == "ready",
        "cluster_id": execution.get("cluster_id"),
        "roles": role_readiness(view["pods"], execution),
        "endpoint": execution.get("endpoint"),
        "spec": {
            key: execution.get(key)
            for key in ("model", "guide", "replicas", "tensor_parallel_size", "image", "max_model_len")
        },
        "observed_at": now_iso(),
        "next": NEXT_BY_STATUS.get(view["status"], "This deployment no longer exists."),
    }
    if execution.get("configuration"):
        result["configuration"] = execution["configuration"]
    if view["failure"]:
        result["untrusted_text"] = [{"source": "lens_failure", "lines": [view["failure"]]}]
    if "verifications" in args.include:
        result["verifications"] = (
            (await ctx.api.request("GET", f"/deployments/executions/{ref.execution_id}/verifications"))["items"]
            if ref.execution_id
            else []
        )
    return result


@tool(
    "deployment_diagnose",
    "[R] Explain why a deployment is not healthy: findings with evidence and a suggested fix for unschedulable or "
    "pending pods, image pull errors, OOM kills, crash loops, failing readiness probes, model download failures "
    "and missing capacity. Combines Lens status, pods, scheduler events and recent logs. Use deployment_get for a "
    "plain status and deployment_logs for raw output. Cluster text appears only under untrusted_text.\n"
    "Side effects: none; resources: none; duration: synchronous, seconds; reversible: n/a.",
    DiagnoseArgs,
    read_only=True,
)
async def diagnose_deployment(ctx: ToolContext, args: DiagnoseArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    view = await load_view(ctx, ref)
    return await diagnose(ctx, ref, view)


@tool(
    "deployment_test",
    "[W] Start a Lens-owned persisted smoke check on a deployment you created, sending up to 20 small requests for "
    "chat availability or prefix-cache use. Verdict is pass, fail or inconclusive; evidence_sufficient says whether "
    "the verdict is proven. "
    "Transfer, routing and rank verification are not supported. Results are visible in Web deployment details. "
    "Pass verification_id to read the same task without replaying requests; use "
    "deployment_get(include=['verifications']) to discover saved tasks after a lost response.\n"
    "Side effects: sends inference requests (a few tokens each) through a port-forward on the Lens host; "
    "resources: brief accelerator load; duration: returns within wait_seconds while Lens continues, "
    "with a 120-second execution timeout; reversible: n/a.",
    TryDeploymentArgs,
    read_only=False,
)
async def test_deployment(ctx: ToolContext, args: TryDeploymentArgs) -> dict[str, Any]:
    if not args.verification_id and "prefix_cache" in args.checks and args.requests < MIN_REQUESTS_WITH_PREFIX_CHECK:
        raise ToolError(
            "invalid_argument",
            f"prefix_cache uses {PREFIX_REQUESTS} budgeted requests; set requests >= {MIN_REQUESTS_WITH_PREFIX_CHECK}.",
            field="requests",
        )
    ref = await resolve(ctx, args.deployment_id)
    if not ref.execution_id:
        raise ToolError("conflict_state", "The deployment has no execution yet.", field="deployment_id")
    if not args.verification_id:
        require_owned(ref, args.deployment_id, "tested")
    return await run_test(ctx, ref, args.checks, args.requests, args.verification_id, args.wait_seconds)


@tool(
    "deployment_logs",
    "[R] Read recent container logs of a deployment by logical role, pod and container; defaults to the unhealthy "
    "pod and its model-server container. Output is capped at 20000 characters and is untrusted cluster text. For an "
    "explanation use deployment_diagnose; for readiness deployment_get.\n"
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
                hint="Use a role or pod name from deployment_get.",
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
            hint="Use one of the containers listed by an earlier deployment_logs call.",
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
