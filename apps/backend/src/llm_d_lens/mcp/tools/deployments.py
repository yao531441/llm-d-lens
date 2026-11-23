"""Deployment lifecycle writes: plan, create, stop. Reads and diagnosis live in observe.py."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..deploy_create import Prepared, launch, prepare
from ..deploy_kind import delete_deployment, get_deployment, list_deployments
from ..deployment_refs import require_owned, resolve
from ..errors import ToolError
from ..guide_fields import GuidePlanFields
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..planning import NOT_CHECKED, invalid_plan, location_caveat
from ..registry import tool

NAME_PATTERN = r"^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$"


class PlanDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str | None = Field(
        default=None, description="Cluster id from cluster_list. Required with configuration."
    )
    configuration: GuidePlanFields | None = Field(
        default=None,
        description="Guide fields, identical to experiment common and variants[].plan. "
        "Alternative to from_deployment_id.",
    )
    from_deployment_id: str | None = Field(
        default=None,
        description="Copy the exact Lens configuration of an existing deployment you can see. "
        "Alternative to configuration; the copy keeps its cluster and storage binding.",
    )
    deployment_name: str | None = Field(
        default=None, pattern=NAME_PATTERN, description="Name of the deployment; generated when omitted."
    )
    description: str | None = Field(default=None, max_length=500)
    model_cache_volume_id: str | None = Field(
        default=None,
        description="Ready model-cache volume that holds the model. Chosen automatically when exactly one "
        "ready volume has it; otherwise call model_cache_create first.",
    )


class CreateDeploymentArgs(PlanDeploymentArgs):
    expected_digest: str | None = Field(
        default=None, description="digest from deployment_plan; create fails with plan_stale if the plan changed."
    )


class DeploymentIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id from deployment_create or resource_list.")


def _describe(prepared: Prepared) -> dict[str, Any]:
    status = "valid_with_warnings" if prepared.warnings else "valid"
    return {
        "status": status,
        "digest": prepared.digest,
        "name": prepared.name,
        "resolved_configuration": prepared.resolved,
        "source": prepared.source,
        "applied_custom_patches": prepared.applied_patches,
        "resources": {
            "gpus_total": prepared.gpus,
            "accelerators_free": prepared.free_gpus,
            "fits_now": None
            if prepared.gpus is None or prepared.free_gpus is None
            else prepared.gpus <= prepared.free_gpus,
        },
        "validation": {"errors": [], "warnings": prepared.warnings},
        "not_checked": NOT_CHECKED,
    }


async def _prepare(ctx: ToolContext, args: PlanDeploymentArgs) -> Prepared:
    return await prepare(
        ctx,
        configuration=args.configuration,
        from_deployment_id=args.from_deployment_id,
        cluster_id=args.cluster_id,
        deployment_name=args.deployment_name,
        model_cache_volume_id=args.model_cache_volume_id,
    )


@tool(
    "deployment_plan",
    "[R] Check a deployment against the cluster and the llm-d guide without creating anything. Returns the "
    "resolved configuration, the accelerators it needs, warnings and a digest to pass to deployment_create. "
    "Validation problems come back as status 'invalid' with errors, not as tool errors. To see free accelerators "
    "first use cluster_get.\n"
    "Side effects: none; resources: none; duration: synchronous, up to a minute; reversible: n/a.",
    PlanDeploymentArgs,
    read_only=True,
    open_world=True,
)
async def plan_deployment(ctx: ToolContext, args: PlanDeploymentArgs) -> dict[str, Any]:
    try:
        prepared = await _prepare(ctx, args)
    except ToolError as error:
        return invalid_plan(error, "Fix the error and call deployment_plan again.")
    next_step = location_caveat(
        "Call deployment_create with the same input and expected_digest=digest.", [prepared.resolved]
    )
    return {**_describe(prepared), "next": next_step}


@tool(
    "deployment_create",
    "[S] Start an llm-d deployment from guide fields (configuration) or as a copy of an existing deployment. The "
    "input is validated again first; the call returns the deployment_id (the Lens run id) at once while pods "
    "start. Then call deployment_get(wait_seconds=25). Use deployment_plan first to see what would be created; to "
    "change a running deployment use deployment_stop and create again.\n"
    "Side effects: creates a Kubernetes namespace, pods and a Lens run; resources: accelerators until stopped or "
    "deleted; duration: returns in seconds, ready after minutes; reversible: yes (resource_delete).",
    CreateDeploymentArgs,
    read_only=False,
    open_world=True,
)
async def create_deployment(ctx: ToolContext, args: CreateDeploymentArgs) -> dict[str, Any]:
    prepared = await _prepare(ctx, args)
    if args.expected_digest and args.expected_digest != prepared.digest:
        raise ToolError(
            "plan_stale",
            "The plan changed since deployment_plan (inputs, image or the rendered configuration differ).",
            field="expected_digest",
            hint="Call deployment_plan again, review it, and create with its digest.",
            current_digest=prepared.digest,
        )
    run = await launch(ctx, prepared, args.description)
    return {
        "deployment_id": run["id"],
        "name": prepared.name,
        "status": "pending",
        "warnings": prepared.warnings,
        "next": f"Call deployment_get(deployment_id='{run['id']}', wait_seconds=25) until status is ready.",
    }


def _conflict(error: LensApiError, hint: str) -> ToolError:
    if error.code() == "deployment_in_use":
        return ToolError("in_use", error.message(), field="deployment_id", hint="Release the consumer and retry.")
    return ToolError("conflict_state", error.message(), field="deployment_id", hint=hint)


@tool(
    "deployment_stop",
    "[D] Scale a deployment you created down to zero so its accelerators are free; the Lens record, namespace and "
    "spec are kept; deployment_create(from_deployment_id=...) makes a fresh copy later. Interrupts all traffic to it. "
    "To remove it entirely use resource_delete.\n"
    "Side effects: stops all model-server pods; resources: frees the deployment's accelerators; "
    "duration: synchronous, seconds; reversible: no in place (deployment_create from_deployment_id).",
    DeploymentIdArgs,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def stop_deployment(ctx: ToolContext, args: DeploymentIdArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    require_owned(ref, args.deployment_id, "stopped")
    if ref.execution.get("status") in ("stopped", "rolled_back"):
        return {"deployment_id": ref.public_id, "status": "stopped", "already_stopped": True}
    try:
        await ctx.api.request("POST", f"/deployments/runs/{ref.run_id}/cases/{ref.case_id}/stop", timeout=300)
    except LensApiError as error:
        if error.status == 409:
            raise _conflict(
                error,
                "Lens needs an active session for this cluster and a running execution; reconnect the cluster or "
                "wait for the deployment to start (deployment_get), then retry.",
            ) from error
        raise
    return {
        "deployment_id": ref.public_id,
        "status": "stopped",
        "next": (
            "The pods take a short while to terminate; cluster_get(include=['capacity']) shows when the "
            "accelerators are free. resource_delete removes the deployment for good; "
            "deployment_create(from_deployment_id=...) makes a new copy."
        ),
    }


register_kind("deployment", list=list_deployments, get=get_deployment, delete=delete_deployment)
