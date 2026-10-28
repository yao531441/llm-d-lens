"""The only place experiments touch deployment creation and deletion.

It builds on ``planning.build_plan(ctx, DeploymentSpec)`` (plan cache payload: ``cluster_id``, ``cluster_session_id``,
``deployable_configuration``, ``name``, ``spec``) and POSTs ``/deployments/runs`` itself, one configuration per run,
like ``tools.deployments.create_deployment`` but without waiting for the execution id and with the ledger row
parented to the experiment. When the deployments module exposes a different creation API, rewire ``to_deployment_spec``,
``plan_spec`` and ``launch_spec`` (the only functions that know it).
"""

from __future__ import annotations

import time
from typing import Any

from .context import ToolContext
from .deploy_spec import DeploymentSpec, EngineSpec, RoleSpec, Topology, gpus_total, parse_spec, plan_params
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError
from .planning import build_plan

CREATED_VIA = "lens-mcp-v2"
DELETE_TIMEOUT_SECONDS = 150  # stays well below the experiment lease TTL
_GONE = {"cleaned", "cleaned_up", "rolled_back", "rolling_back"}


def spec_gpus(spec: dict[str, Any]) -> int:
    """Accelerators the spec occupies: decode/serving pool plus the prefill pool of pd-disaggregation."""
    decode = (spec.get("replicas") or 1) * (spec.get("tensor_parallel_size") or 1)
    if spec.get("guide") == "pd-disaggregation":
        decode += (spec.get("prefill_replicas") or 1) * (spec.get("prefill_tensor_parallel_size") or 1)
    return decode


def to_deployment_spec(cluster_id: str, spec: dict[str, Any]) -> DeploymentSpec:
    """The experiment's flat variant spec (plan_deployment-style fields) as a ``DeploymentSpec``."""
    decode = RoleSpec(replicas=spec.get("replicas") or 1, tensor_parallel_size=spec.get("tensor_parallel_size") or 1)
    if spec.get("guide") == "pd-disaggregation":
        prefill = RoleSpec(
            replicas=spec.get("prefill_replicas") or 1,
            tensor_parallel_size=spec.get("prefill_tensor_parallel_size") or 1,
        )
        topology = Topology(prefill=prefill, decode=decode)
    else:
        topology = Topology(serving=decode)
    return DeploymentSpec(
        cluster_id=cluster_id,
        model=spec["model"],
        guide=spec["guide"],
        accelerator=spec.get("accelerator"),
        engine_kind=spec.get("model_server"),
        variant=spec.get("variant"),
        topology=topology,
        engine=EngineSpec(max_model_len=spec.get("max_model_len")),
        runtime_image=spec.get("image"),
    )


def row_gpus(row: Resource) -> int:
    """Accelerators of an existing deployment from its stored spec; 0 when unknown."""
    try:
        return gpus_total(plan_params(parse_spec(row.meta.get("spec") or {})))
    except ToolError:
        return 0


async def plan_spec(ctx: ToolContext, cluster_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    return await build_plan(ctx, to_deployment_spec(cluster_id, spec))


async def launch_spec(
    ctx: ToolContext, *, cluster_id: str, spec: dict[str, Any], root_id: str, variant_id: str, expires_at: float
) -> Resource:
    """Plan and start one temporary deployment; its ledger row is written first and rolled back on failure."""
    plan = await plan_spec(ctx, cluster_id, spec)
    payload = ctx.ledger.get_plan(plan["plan_id"], ctx.owner)
    row = ctx.ledger.add_resource(
        kind="deployment",
        owner=ctx.owner,
        cluster_id=payload["cluster_id"],
        name=payload["name"],
        root_id=root_id,
        state="creating",
        meta={
            "spec": payload["spec"],
            "experiment_id": root_id,
            "variant_id": variant_id,
            "expires_at": expires_at,
            "launch_epoch": time.time(),
        },
    )
    try:
        run = await ctx.api.request(
            "POST",
            "/deployments/runs",
            body={
                "configurations": [payload["deployable_configuration"]],
                "provenance": {
                    "cluster_session_id": payload["cluster_session_id"],
                    "cluster_server_id": payload["cluster_id"],
                    "created_via": CREATED_VIA,
                    "ledger_id": row.id,
                    "experiment_id": root_id,
                },
            },
            timeout=120,
        )
    except LensApiError as error:
        if 400 <= error.status < 500 and error.status != 408:
            ctx.ledger.set_state(row.id, "deleted")
            raise
        raise LaunchUncertainError(row.id) from error
    except Exception as error:
        # Timeout or transport failure: Lens may have accepted the request, so the row stays "creating" for
        # reconciliation by ledger_id. CancelledError is a BaseException and propagates the same way.
        raise LaunchUncertainError(row.id) from error
    ctx.ledger.bind_run(row.id, run["id"])
    ctx.ledger.set_state(row.id, "active")
    ctx.ledger.drop_plan(plan["plan_id"])
    return ctx.ledger.get(row.id)  # type: ignore[return-value]


class LaunchUncertainError(Exception):
    """The create request may or may not have reached Lens; the ledger row (``row_id``) stays ``creating``."""

    def __init__(self, row_id: str):
        super().__init__(f"Creation of {row_id} is unconfirmed.")
        self.row_id = row_id


async def adopt_created_run(ctx: ToolContext, row: Resource) -> bool:
    """Bind the Lens run that carries this row's ``ledger_id`` provenance; ``False`` when none is visible (yet)."""
    runs = await ctx.api.request("GET", "/deployments/runs")
    run = next(
        (
            item
            for item in runs or []
            if isinstance(item, dict) and (item.get("provenance") or {}).get("ledger_id") == row.id
        ),
        None,
    )
    if run is None:
        return False
    ctx.ledger.bind_run(row.id, run["id"])
    ctx.ledger.set_state(row.id, "active")
    return True


async def _execution_id(ctx: ToolContext, row: Resource) -> str | None:
    if row.lens_execution_id or not row.lens_run_id:
        return row.lens_execution_id
    try:
        run = await ctx.api.request("GET", f"/deployments/runs/{row.lens_run_id}")
    except LensApiError as error:
        if error.status == 404:
            return None
        raise
    execution_id = next((case["execution_id"] for case in run.get("cases") or [] if case.get("execution_id")), None)
    if execution_id:
        ctx.ledger.bind_execution(row.id, execution_id)
    return execution_id


async def deployment_state(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    """``state`` is one of starting, ready, failed, gone."""
    execution_id = await _execution_id(ctx, row)
    if execution_id is None:
        return {"state": "starting", "execution_id": None, "failure": await _run_failure(ctx, row)}
    try:
        execution = await ctx.api.request("GET", f"/deployments/executions/{execution_id}")
    except LensApiError as error:
        if error.status == 404:
            return {"state": "gone", "execution_id": execution_id}
        raise
    status = execution.get("status")
    state = (
        "ready" if status == "ready" else "failed" if status == "failed" else "gone" if status in _GONE else "starting"
    )
    return {
        "state": state,
        "execution_id": execution_id,
        "endpoint": execution.get("endpoint"),
        "failure": execution.get("failure") if state == "failed" else None,
    }


async def _run_failure(ctx: ToolContext, row: Resource) -> Any:
    if not row.lens_run_id:
        return None
    try:
        run = await ctx.api.request("GET", f"/deployments/runs/{row.lens_run_id}")
    except LensApiError:
        return None
    return next((case["failure"] for case in run.get("cases") or [] if case.get("failure")), None)


async def delete_deployment(ctx: ToolContext, row: Resource) -> bool:
    """Tear the deployment down; ``False`` means Lens still sees it in use (an evaluation is active)."""
    execution_id = await _execution_id(ctx, row)
    try:
        if execution_id:
            await ctx.api.request(
                "DELETE",
                f"/deployments/executions/{execution_id}",
                params={"delete_namespace": True},
                timeout=DELETE_TIMEOUT_SECONDS,
            )
        elif row.lens_run_id:
            await ctx.api.request("DELETE", f"/deployments/runs/{row.lens_run_id}", timeout=DELETE_TIMEOUT_SECONDS)
    except LensApiError as error:
        if error.status == 409 and error.code() == "deployment_in_use":
            return False
        if error.status != 404:
            raise
    ctx.ledger.set_state(row.id, "deleted")
    return True


async def borrowed_target(ctx: ToolContext, deployment_id: str) -> tuple[Resource, dict[str, Any]]:
    """An existing deployment the caller owns; returns its ledger row and live state. Never modified later."""
    row = ctx.ledger.get(deployment_id)
    if row is None or row.kind != "deployment" or row.state == "deleted":
        raise ToolError(
            "owner_unknown",
            f"'{deployment_id}' is not a deployment created through this server, so it cannot be load-tested.",
            field="deployment_id",
            hint="Use a spec variant to let the experiment create its own deployment, or pass a dep- id from "
            "list_resources.",
        )
    if row.owner != ctx.owner:
        raise ToolError(
            "not_owner",
            f"{deployment_id} belongs to another user; benchmarking adds load to it, so it is refused.",
            field="deployment_id",
            hint="Deploy your own copy with a spec variant.",
        )
    state = await deployment_state(ctx, row)
    if state["state"] != "ready":
        raise ToolError(
            "conflict_state",
            f"{deployment_id} is {state['state']}, not ready.",
            field="deployment_id",
            hint="Wait until get_deployment reports ready=true, then retry.",
        )
    return row, state
