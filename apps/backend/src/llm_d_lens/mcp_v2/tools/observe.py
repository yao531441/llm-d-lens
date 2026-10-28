"""Read-only views of deployments: list, status, logs. Work for any deployment the Lens user may read."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..deployment_refs import DeploymentRef, not_found, resolve
from ..lens_api import LensApiError
from ..registry import tool

MAX_LOG_CHARS = 20000


class ListDeploymentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str | None = Field(default=None, description="Only deployments on this cluster.")
    mine_only: bool = Field(default=True, description="Only deployments you created through create_deployment.")


class GetDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id from create_deployment or list_deployments.")


class GetLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    pod: str | None = Field(
        default=None, description="Pod name from get_deployment. Defaults to a pod that is not ready."
    )
    container: str | None = None
    tail: int = Field(default=200, ge=1, le=2000, description="Number of trailing log lines.")
    previous: bool = Field(default=False, description="Logs of the previous container instance after a restart.")


def _summary(execution: dict[str, Any], deployment_id: str, owned: bool) -> dict[str, Any]:
    return {
        "deployment_id": deployment_id,
        "owned": owned,
        "name": execution.get("name"),
        "status": execution.get("status"),
        "ready": execution.get("status") == "ready",
        "cluster_id": execution.get("cluster_id"),
        "namespace": execution.get("namespace"),
        "model": execution.get("model"),
        "guide": execution.get("guide"),
        "replicas": execution.get("replicas"),
        "tensor_parallel_size": execution.get("tensor_parallel_size"),
        "created_at": execution.get("created_at"),
    }


@tool(
    "list_deployments",
    "List deployments with status. By default only the ones you created; set mine_only=false to see everyone's "
    "(read-only: you can inspect them but not delete them).",
    ListDeploymentsArgs,
    read_only=True,
)
async def list_deployments(ctx: ToolContext, args: ListDeploymentsArgs) -> dict[str, Any]:
    rows = ctx.ledger.list_resources(owner=ctx.owner, kind="deployment", cluster_id=args.cluster_id)
    mine = {row.lens_execution_id: row for row in rows if row.lens_execution_id}
    data = await ctx.api.request("GET", "/deployments/executions", params={"cluster_id": args.cluster_id, "limit": 500})
    items, seen = [], set()
    for execution in data.get("items") or []:
        row = mine.get(execution["execution_id"])
        if args.mine_only and row is None:
            continue
        seen.add(execution["execution_id"])
        items.append(_summary(execution, row.id if row else execution["execution_id"], row is not None))
    for row in rows:
        if row.lens_execution_id not in seen:
            items.append(
                {
                    "deployment_id": row.id,
                    "owned": True,
                    "name": row.name,
                    "status": "pending",
                    "ready": False,
                    "cluster_id": row.cluster_id,
                }
            )
    return {"deployments": items}


async def _pods(ctx: ToolContext, execution_id: str) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", f"/deployments/executions/{execution_id}/pods", timeout=30)
    return [
        {
            "name": pod.get("name"),
            "phase": pod.get("phase"),
            "ready": pod.get("ready"),
            "restarts": pod.get("restarts"),
            "reason": pod.get("status_reason"),
            "node": pod.get("node_name"),
            "containers": [c.get("name") for c in pod.get("containers") or []],
        }
        for pod in data.get("pods") or []
    ]


def _hint(status: str | None, pods: list[dict[str, Any]]) -> str | None:
    if status == "failed":
        return "Deployment failed; call get_logs to see why, then delete_deployment."
    if any(pod["phase"] == "Pending" for pod in pods):
        return "A pod is Pending: the cluster likely lacks free accelerators. Check get_cluster."
    if any((pod["restarts"] or 0) > 2 for pod in pods):
        return "A pod is restarting; call get_logs with previous=true."
    if status != "ready":
        return "Still starting; model download and warm-up can take several minutes. Poll again shortly."
    return None


async def _execution(ctx: ToolContext, ref: DeploymentRef, requested: str) -> dict[str, Any] | None:
    if not ref.execution_id:
        return None
    try:
        return await ctx.api.request("GET", f"/deployments/executions/{ref.execution_id}")
    except LensApiError as error:
        if error.status != 404:
            raise
        if ref.row is None:
            raise not_found(requested) from error
        return None


@tool(
    "get_deployment",
    "Show a deployment's status, readiness, endpoint and pods, with a hint for the next step. Poll this after "
    "create_deployment until ready is true.",
    GetDeploymentArgs,
    read_only=True,
)
async def get_deployment(ctx: ToolContext, args: GetDeploymentArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    execution = await _execution(ctx, ref, args.deployment_id)
    if execution is None:
        if ref.row is not None and ref.row.state == "deleted":
            return {"deployment_id": ref.row.id, "owned": True, "status": "deleted", "ready": False}
        run = await ctx.api.request("GET", f"/deployments/runs/{ref.run_id}") if ref.run_id else {}
        failure = next((case.get("failure") for case in run.get("cases") or [] if case.get("failure")), None)
        return {
            "deployment_id": ref.public_id,
            "owned": ref.owned,
            "status": run.get("status", "pending"),
            "ready": False,
            "failure": failure,
            "hint": "Creation failed before pods were started; delete_deployment and replan."
            if failure
            else "Deployment is still being scheduled; poll again in a few seconds.",
        }
    pods = await _pods(ctx, ref.execution_id) if execution.get("namespace") else []
    result = _summary(execution, ref.public_id, ref.owned)
    result.update(
        {
            "image": execution.get("image"),
            "endpoint": execution.get("endpoint"),
            "failure": execution.get("failure"),
            "pods": pods,
            "hint": _hint(execution.get("status"), pods),
        }
    )
    return result


@tool(
    "get_logs",
    "Fetch recent log lines from one pod of a deployment. Use it when a deployment is failed, restarting or "
    "stuck starting.",
    GetLogsArgs,
    read_only=True,
)
async def get_logs(ctx: ToolContext, args: GetLogsArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    if not ref.execution_id:
        raise not_found(args.deployment_id)
    pods = await _pods(ctx, ref.execution_id)
    pod = args.pod
    if pod is None:
        candidates = [p for p in pods if not p["ready"]] or pods
        if not candidates:
            return {"deployment_id": ref.public_id, "pods": [], "logs": "", "note": "No pods exist yet."}
        pod = candidates[0]["name"]
    elif pod not in {p["name"] for p in pods}:
        raise not_found(f"{args.deployment_id}/{pod}")
    data = await ctx.api.request(
        "GET",
        f"/deployments/executions/{ref.execution_id}/pods/{pod}/logs",
        params={"container": args.container, "tail": args.tail, "previous": args.previous},
        timeout=40,
    )
    logs = data.get("logs") or ""
    return {
        "deployment_id": ref.public_id,
        "pod": pod,
        "container": data.get("container"),
        "success": data.get("success"),
        "truncated": len(logs) > MAX_LOG_CHARS,
        "logs": logs[-MAX_LOG_CHARS:],
        "other_pods": [p["name"] for p in pods if p["name"] != pod],
    }
