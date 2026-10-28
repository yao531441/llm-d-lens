"""Deployment lifecycle writes: plan, create, update, stop, restart. Reads and diagnosis live in observe.py."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .. import deploy_kind
from ..context import ToolContext
from ..deploy_create import DEFAULT_TTL_HOURS, expires_at, launch_plan
from ..deploy_recommend import recommend
from ..deploy_spec import DeploymentSpec, spec_from_deployment
from ..deploy_status import iso
from ..deployment_refs import get_run, latest_case, require_owned, resolve
from ..errors import ToolError
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..planning import build_plan
from ..registry import tool

LABEL_KEY = r"^[A-Za-z0-9][A-Za-z0-9_.\-/]{0,62}$"


class PlanDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["validate", "recommend"] = Field(
        default="validate",
        description="'validate' checks and renders one spec and returns a plan_id. 'recommend' ranks guide / "
        "replica / tensor-parallel choices for a model (spec needs only cluster_id and model).",
    )
    spec: DeploymentSpec | None = Field(
        default=None, description="Full deployment spec. Alternative to from_deployment_id."
    )
    from_deployment_id: str | None = Field(
        default=None, pattern=r"^dep-[0-9a-f]+$", description="Start from the stored spec of one of your deployments."
    )
    spec_patch: dict[str, Any] | None = Field(
        default=None,
        description="JSON merge patch (RFC 7396) applied to the stored spec, for example "
        '{"topology": {"serving": {"replicas": 2}}}. Only with from_deployment_id.',
    )


class CreateDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: DeploymentSpec | None = Field(
        default=None, description="Full spec; validated the same way as plan_deployment."
    )
    from_deployment_id: str | None = Field(
        default=None,
        pattern=r"^dep-[0-9a-f]+$",
        description="Clone one of your deployments, optionally with spec_patch.",
    )
    spec_patch: dict[str, Any] | None = Field(
        default=None, description="JSON merge patch applied to the stored spec. Only with from_deployment_id."
    )
    plan_id: str | None = Field(default=None, description="plan_id from plan_deployment (valid for one hour).")
    ttl_hours: int = Field(
        default=DEFAULT_TTL_HOURS, ge=1, le=72, description="Hours until the deployment is due for cleanup (1-72)."
    )
    client_request_id: str | None = Field(
        default=None, max_length=64, description="Retry key: repeating it returns the first result."
    )


class UpdateDeploymentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) from create_deployment or list_resources.")
    name: str | None = Field(default=None, pattern=r"^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$", description="New name.")
    description: str | None = Field(default=None, max_length=500, description="New description.")
    ttl_hours: int | None = Field(
        default=None, ge=1, le=72, description="Restart the TTL clock: now + this many hours."
    )
    labels: dict[str, str] | None = Field(
        default=None, max_length=20, description="Replaces all labels (kept by this server; {} clears them)."
    )
    monitoring: bool | None = Field(default=None, description="true enables, false disables metrics scraping.")


class DeploymentIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(description="Deployment id (dep-...) from create_deployment or list_resources.")


def _spec_from(
    ctx: ToolContext, spec: DeploymentSpec | None, from_id: str | None, patch: dict | None
) -> DeploymentSpec:
    if spec is not None and from_id:
        raise ToolError("invalid_argument", "Pass spec or from_deployment_id, not both.", field="spec")
    if patch is not None and not from_id:
        raise ToolError("invalid_argument", "spec_patch needs from_deployment_id.", field="spec_patch")
    if spec is not None:
        return spec
    if from_id:
        return spec_from_deployment(ctx, from_id, patch)
    raise ToolError(
        "invalid_argument",
        "Provide spec or from_deployment_id.",
        field="spec",
        hint="See list_guides, list_models and get_cluster for the values to put in a spec.",
    )


@tool(
    "plan_deployment",
    "[R] Check a deployment spec against the cluster and the llm-d guide without creating anything, and return a "
    "plan_id for create_deployment; or, with mode='recommend', rank guide / replica / tensor-parallel choices for a "
    "model. Validation problems come back as status 'invalid' with errors, not as tool errors. To see free "
    "accelerators first use get_cluster.\n"
    "Side effects: stores a plan for one hour; resources: none reserved (capacity is not held); "
    "duration: synchronous, up to a minute; reversible: n/a. Recommendations are heuristics on free capacity, not "
    "performance predictions.",
    PlanDeploymentArgs,
    read_only=True,
    open_world=True,
)
async def plan_deployment(ctx: ToolContext, args: PlanDeploymentArgs) -> dict[str, Any]:
    if args.mode == "recommend":
        if args.spec is None or args.from_deployment_id or args.spec_patch:
            raise ToolError(
                "invalid_argument", "mode='recommend' takes only spec (cluster_id and model).", field="spec"
            )
        return await recommend(ctx, args.spec)
    spec = _spec_from(ctx, args.spec, args.from_deployment_id, args.spec_patch)
    try:
        return await build_plan(ctx, spec)
    except ToolError as error:
        problem = {"code": error.code, "message": error.message, "field": error.field, "hint": error.hint}
        return {
            "status": "invalid",
            "validation": {"errors": [problem], "warnings": error.extra.get("warnings") or []},
            "resolved_spec": spec.model_dump(exclude_none=True),
            "next": "Fix the error and call plan_deployment again.",
        }


@tool(
    "create_deployment",
    "[S] Start an llm-d deployment from a spec, from an existing deployment plus spec_patch, or from a plan_id. "
    "The spec is validated synchronously first; the call returns the deployment_id at once while pods start. Then "
    "call get_deployment(wait_seconds=25). Use plan_deployment first to see what would be created; to change a "
    "running deployment use stop_deployment and create again.\n"
    "Side effects: creates a Kubernetes namespace, pods and a Lens run; resources: accelerators until stopped or "
    "deleted; duration: returns in seconds, ready after minutes; reversible: yes (delete_resources).",
    CreateDeploymentArgs,
    read_only=False,
    open_world=True,
)
async def create_deployment(ctx: ToolContext, args: CreateDeploymentArgs) -> dict[str, Any]:
    given = [name for name in ("spec", "from_deployment_id", "plan_id") if getattr(args, name)]
    if len(given) != 1:
        raise ToolError(
            "invalid_argument",
            "Provide exactly one of spec, from_deployment_id or plan_id.",
            field="spec",
            hint="plan_id comes from plan_deployment.",
        )
    warnings: list[Any] = []
    plan_id = args.plan_id
    if plan_id is None:
        plan = await build_plan(ctx, _spec_from(ctx, args.spec, args.from_deployment_id, args.spec_patch))
        plan_id, warnings = plan["plan_id"], plan["validation"]["warnings"]
    elif args.spec_patch is not None:
        raise ToolError("invalid_argument", "spec_patch needs from_deployment_id.", field="spec_patch")
    row = await launch_plan(ctx, plan_id, ttl_hours=args.ttl_hours)
    return {
        "deployment_id": row.id,
        "name": row.name,
        "status": "pending",
        "expires_at": iso(row.meta["expires_at"]),
        "warnings": warnings,
        "next": f"Call get_deployment(deployment_id='{row.id}', wait_seconds=25) until status is ready.",
    }


@tool(
    "update_deployment",
    "[W] Change a deployment you created: name, description, TTL (restarted from now, 1-72 hours), labels, or the "
    "monitoring flag. It cannot change the topology or model; for that stop_deployment and create_deployment "
    "(from_deployment_id + spec_patch).\n"
    "Side effects: edits Lens and ledger metadata; monitoring toggles metrics scraping; resources: none; "
    "duration: synchronous; reversible: yes.",
    UpdateDeploymentArgs,
    read_only=False,
    idempotent=True,
)
async def update_deployment(ctx: ToolContext, args: UpdateDeploymentArgs) -> dict[str, Any]:
    changes = {key: getattr(args, key) for key in ("name", "description", "ttl_hours", "labels", "monitoring")}
    changes = {key: value for key, value in changes.items() if value is not None}
    if not changes:
        raise ToolError(
            "invalid_argument",
            "Nothing to update.",
            field="deployment_id",
            hint="Pass at least one of name, description, ttl_hours, labels, monitoring.",
        )
    ref = await resolve(ctx, args.deployment_id)
    row = require_owned(ref, args.deployment_id, "updated")
    remote = {key: value for key, value in changes.items() if key in ("name", "description")}
    if (remote or "monitoring" in changes) and not ref.execution_id:
        raise ToolError(
            "conflict_state",
            "The deployment has no running execution yet.",
            field="deployment_id",
            hint="Call get_deployment(wait_seconds=25) first, or update only ttl_hours / labels.",
        )
    if remote:
        patch = {"display_name": remote["name"]} if "name" in remote else {}
        if "description" in remote:
            patch["description"] = remote["description"]
        await ctx.api.request("PATCH", f"/deployments/executions/{ref.execution_id}", body=patch)
    monitoring = None
    if "monitoring" in changes:
        action = "enable" if changes["monitoring"] else "disable"
        result = await ctx.api.request(
            "POST", f"/monitoring/deployments/{ref.execution_id}/{action}", params={"cluster_id": row.cluster_id}
        )
        monitoring = (result or {}).get("status") or action + "d"
    meta: dict[str, Any] = {}
    if "ttl_hours" in changes:
        meta["expires_at"] = expires_at(changes["ttl_hours"])
    if "labels" in changes:
        meta["labels"] = changes["labels"]
    if meta:
        ctx.ledger.update_meta(row.id, meta)
    if "name" in changes:
        ctx.ledger.set_name(row.id, changes["name"])
    current = ctx.ledger.get(row.id)
    return {
        "deployment_id": row.id,
        "updated": sorted(changes),
        "name": current.name,
        "expires_at": iso(current.meta["expires_at"]) if current.meta.get("expires_at") else None,
        "labels": current.meta.get("labels") or {},
        **({"monitoring": monitoring} if monitoring else {}),
    }


async def _case(ctx: ToolContext, run_id: str | None) -> dict[str, Any]:
    case = latest_case(await get_run(ctx, run_id) or {}) if run_id else None
    if not case or not case.get("id"):
        raise ToolError(
            "conflict_state",
            "Lens has no deployment case for this deployment yet.",
            field="deployment_id",
            hint="Call get_deployment(wait_seconds=25) and retry.",
        )
    return case


def _conflict(error: LensApiError, hint: str) -> ToolError:
    return ToolError("conflict_state", error.message(), field="deployment_id", hint=hint)


@tool(
    "stop_deployment",
    "[D] Scale a deployment you created down to zero so its accelerators are free; the Lens record, namespace and "
    "spec are kept and restart_deployment brings it back. Interrupts all traffic to it. To remove it entirely use "
    "delete_resources.\n"
    "Side effects: stops all model-server pods; resources: frees the deployment's accelerators; "
    "duration: synchronous, seconds; reversible: yes (restart_deployment).",
    DeploymentIdArgs,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def stop_deployment(ctx: ToolContext, args: DeploymentIdArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    row = require_owned(ref, args.deployment_id, "stopped")
    if users := await deploy_kind.borrowers(ctx, row):
        raise deploy_kind.borrowed_error(row.id, users)
    case = await _case(ctx, ref.run_id)
    if case.get("status") == "stopped":
        return {"deployment_id": row.id, "status": "stopped", "already_stopped": True}
    try:
        await ctx.api.request("POST", f"/deployments/runs/{ref.run_id}/cases/{case['id']}/stop", timeout=300)
    except LensApiError as error:
        if error.status == 409:
            raise _conflict(
                error,
                "Lens needs an active session for this cluster and a running execution; reconnect the cluster or "
                "wait for the deployment to start (get_deployment), then retry.",
            ) from error
        raise
    return {
        "deployment_id": row.id,
        "status": "stopped",
        "next": "delete_resources removes it for good; create_deployment(from_deployment_id=...) makes a new copy.",
    }


@tool(
    "restart_deployment",
    "[S] Bring a stopped or failed deployment you created back up. Currently unavailable: Lens' restart cleans the "
    "namespace and then rejects its own retry, so this tool refuses without touching anything. Use "
    "create_deployment(from_deployment_id=...) for a fresh copy of the same spec instead, after stop_deployment "
    "or delete_resources on the old one.\n"
    "Side effects: none while Lens restart is broken; resources: none; duration: synchronous; reversible: n/a.",
    DeploymentIdArgs,
    read_only=False,
    idempotent=True,
)
async def restart_deployment(ctx: ToolContext, args: DeploymentIdArgs) -> dict[str, Any]:
    ref = await resolve(ctx, args.deployment_id)
    row = require_owned(ref, args.deployment_id, "restarted")
    case = await _case(ctx, ref.run_id)
    if case.get("status") not in ("stopped", "failed"):
        raise ToolError(
            "conflict_state",
            f"Only stopped or failed deployments can be restarted; this one is {case.get('status')}.",
            field="deployment_id",
            hint="Call stop_deployment first, or get_deployment to see its status.",
        )
    # Lens (deploy/worker.py restart_case) marks the case CLEANED and then asks retry_case, which only accepts
    # FAILED or STOPPED cases, so every restart would destroy the namespace and then fail.
    raise ToolError(
        "conflict_state",
        "Lens cannot restart this deployment: its restart removes the namespace and then rejects the retry.",
        field="deployment_id",
        hint=f"Call create_deployment(from_deployment_id='{row.id}') for a new deployment with the same spec "
        "(delete the old one with delete_resources when it is no longer needed).",
    )


register_kind(
    "deployment",
    delete=deploy_kind.delete,
    describe=deploy_kind.describe,
    dependents=deploy_kind.dependents,
    list_others=deploy_kind.list_others,
)
