"""Deployment lifecycle: plan (read-only), create, delete (own resources only)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..deployment_refs import resolve, wait_for_execution
from ..errors import ToolError
from ..lens_api import LensApiError
from ..planning import build_plan
from ..registry import tool

CREATED_VIA = "lens-mcp-v2"


class PlanDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(description="Cluster id from list_clusters.")
    guide: str = Field(description="llm-d guide id from list_guides, e.g. 'optimized-baseline'.")
    model: str = Field(description="Hugging Face model id, e.g. 'Qwen/Qwen3-0.6B'.")
    accelerator: str | None = Field(
        default=None, description="'xpu' or 'gpu'. Inferred from the cluster when it has exactly one family."
    )
    model_server: str | None = Field(
        default=None, description="'vllm' or 'sglang'. Defaults to vllm when the guide offers it."
    )
    variant: str | None = Field(default=None, description="Guide variant. Defaults to 'base'.")
    replicas: int = Field(
        default=1, ge=1, le=64, description="Model-server replicas (decode replicas for pd-disaggregation)."
    )
    tensor_parallel_size: int = Field(default=1, ge=1, le=64, description="Accelerators per replica.")
    prefill_replicas: int = Field(default=1, ge=1, le=64, description="pd-disaggregation only.")
    prefill_tensor_parallel_size: int = Field(default=1, ge=1, le=64, description="pd-disaggregation only.")
    max_model_len: int | None = Field(default=None, ge=128, description="vLLM context length override.")
    image: str | None = Field(
        default=None, description="Override the model-server image. Defaults to the guide's image."
    )
    name: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$",
        description="Deployment name. Generated when omitted.",
    )


class CreateDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="plan_id returned by plan_deployment.")


class DeleteDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) created through create_deployment.")


@tool(
    "plan_deployment",
    "Validate and render a deployment of an llm-d guide for a model on a cluster. Changes nothing on the cluster. "
    "Returns a plan_id plus what would be deployed and any warnings; pass plan_id to create_deployment.",
    PlanDeploymentArgs,
    read_only=True,
    open_world=True,
)
async def plan_deployment(ctx: ToolContext, args: PlanDeploymentArgs) -> dict[str, Any]:
    return await build_plan(ctx, args)


@tool(
    "create_deployment",
    "Start the deployment described by a plan_id. Returns immediately with a deployment_id; poll get_deployment "
    "until ready. Consumes cluster accelerators until deleted.",
    CreateDeploymentArgs,
    read_only=False,
    open_world=True,
)
async def create_deployment(ctx: ToolContext, args: CreateDeploymentArgs) -> dict[str, Any]:
    plan = ctx.ledger.get_plan(args.plan_id, ctx.owner)
    if plan is None:
        raise ToolError(
            "plan_not_found",
            f"Plan '{args.plan_id}' does not exist or has expired.",
            field="plan_id",
            hint="Call plan_deployment again to get a fresh plan_id.",
        )
    description = plan["description"]
    row = ctx.ledger.add_resource(
        kind="deployment",
        owner=ctx.owner,
        cluster_id=plan["cluster_id"],
        name=description["deployment_name"],
        state="creating",
        meta={"plan": description},
    )
    try:
        session = await ctx.api.request("GET", "/cluster/session", params={"clusterId": plan["cluster_id"]})
        session_id = session["sessionId"]
        run = await ctx.api.request(
            "POST",
            "/deployments/runs",
            body={
                "configurations": [plan["deployable_configuration"]],
                "provenance": {
                    "cluster_session_id": session_id,
                    "cluster_server_id": plan["cluster_id"],
                    "created_via": CREATED_VIA,
                    "ledger_id": row.id,
                    "plan_id": args.plan_id,
                },
            },
            timeout=120,
        )
    except BaseException:
        ctx.ledger.set_state(row.id, "deleted")
        raise
    ctx.ledger.bind_run(row.id, run["id"])
    ctx.ledger.set_state(row.id, "active")
    ctx.ledger.drop_plan(args.plan_id)
    execution_id = await wait_for_execution(ctx, run["id"])
    if execution_id:
        ctx.ledger.bind_execution(row.id, execution_id)
    return {
        "deployment_id": row.id,
        "name": description["deployment_name"],
        "status": run.get("status", "queued"),
        "next": "Poll get_deployment until ready is true; use get_logs if pods are not progressing.",
    }


async def _delete_in_lens(ctx: ToolContext, execution_id: str | None, run_id: str | None) -> None:
    try:
        if execution_id:
            await ctx.api.request(
                "DELETE", f"/deployments/executions/{execution_id}", params={"delete_namespace": True}, timeout=300
            )
        elif run_id:
            await ctx.api.request("DELETE", f"/deployments/runs/{run_id}", timeout=300)
    except LensApiError as error:
        if error.status != 404:
            raise


@tool(
    "delete_deployment",
    "Tear down a deployment you created with create_deployment and free its accelerators. Refuses deployments "
    "that were not created through this tool, and deployments an evaluation is still using.",
    DeleteDeploymentArgs,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def delete_deployment(ctx: ToolContext, args: DeleteDeploymentArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    if ref.row is not None and ref.row.state == "deleted":
        return {"deployment_id": ref.row.id, "deleted": True, "note": "Already deleted."}
    if not ref.owned:
        raise ToolError(
            "not_owned",
            f"Deployment '{args.deployment_id}' was not created through this tool by your Lens user.",
            field="deployment_id",
            hint=(
                "Only deployments you created with create_deployment can be deleted here. "
                "Ask its owner or an administrator."
            ),
        )
    ctx.ledger.set_state(ref.row.id, "deleting")
    try:
        await _delete_in_lens(ctx, ref.execution_id, ref.run_id)
    except LensApiError:
        ctx.ledger.set_state(ref.row.id, "active")
        raise
    ctx.ledger.set_state(ref.row.id, "deleted")
    return {"deployment_id": ref.row.id, "deleted": True}
