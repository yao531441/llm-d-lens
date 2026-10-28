"""Writes on Lens Evaluation workflows: start, cancel, retry, delete and listing. No local state is kept."""

from __future__ import annotations

import asyncio
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .evaluation_design import client_name, evaluation_request
from .evaluation_plan import MAX_LISTED_CASES, Prepared, eta, prepare
from .lens_api import LensApiError
from .path_ids import check_id, is_path_id
from .planning import slug
from .registry import upstream_error

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
WORKFLOWS = "/evaluate/workflow-runs"
CAPACITY_WORDS = ("capacity", "accelerator", "gpu", "xpu")
MCP_PLAN_PREFIX = "lens-mcp"
LIST_LIMIT = 100


async def load_workflow(ctx: ToolContext, experiment_id: str) -> dict[str, Any]:
    check_id(experiment_id, "experiment_id")
    try:
        return await ctx.api.request("GET", f"{WORKFLOWS}/{experiment_id}", timeout=30)
    except LensApiError as error:
        if error.status in (404, 422):
            raise ToolError(
                "not_found",
                f"No experiment '{experiment_id}'.",
                field="experiment_id",
                hint="Call list_resources(kinds=['experiment']) for ids.",
            ) from error
        raise


def guard_owner(ctx: ToolContext, workflow: dict[str, Any]) -> None:
    """Lens checks the route permission but not the record owner, so writes compare it here."""
    owner = workflow.get("owner_user_id")
    if owner is None and ctx.owner == "system":
        return
    if not owner:
        raise ToolError(
            "owner_unknown",
            f"Lens did not record an owner for experiment {workflow.get('id')}.",
            field="experiment_id",
            hint="It is readable but cannot be changed through this server; use the Lens UI.",
        )
    if owner != ctx.owner:
        raise ToolError(
            "not_owner",
            f"Experiment {workflow.get('id')} belongs to another user.",
            field="experiment_id",
            hint="You can read it with get_experiment but not change it.",
        )


def _started(workflow: dict[str, Any], prepared: Prepared | None, artifact_ids: dict[str, str]) -> dict[str, Any]:
    cases = workflow.get("cases") or []
    result: dict[str, Any] = {
        "experiment_id": workflow["id"],
        "status": workflow.get("status"),
        "cases_total": len(cases),
        "artifact_ids": artifact_ids,
        "poll_after_seconds": 60,
        "next": f"Call get_experiment(experiment_id='{workflow['id']}', wait_seconds=30) to follow it.",
    }
    if prepared is not None:
        result.update(
            case_order=prepared.cases[:MAX_LISTED_CASES],
            seed=prepared.seed,
            eta_seconds=eta(prepared),
        )
    return result


async def _existing(ctx: ToolContext, name: str) -> dict[str, Any] | None:
    data = await ctx.api.request("GET", WORKFLOWS, timeout=60)
    return next(
        (
            item
            for item in data.get("items") or []
            if item.get("name") == name and item.get("owner_user_id") in (ctx.owner, None)
        ),
        None,
    )


async def _discard(ctx: ToolContext, artifact_ids: list[str]) -> list[str]:
    """Best-effort removal of artifacts saved for a workflow that was never created; returns the leftovers."""
    leftovers = []
    for artifact_id in artifact_ids:
        try:
            await ctx.api.request("DELETE", f"/configurations/artifacts/{artifact_id}", timeout=30)
        except LensApiError as error:
            if error.status != 404:
                leftovers.append(artifact_id)
        except (TimeoutError, OSError):
            leftovers.append(artifact_id)
    return leftovers


async def _save_artifacts(ctx: ToolContext, prepared: Prepared) -> dict[str, str]:
    saved: dict[str, str] = {}
    try:
        for variant_id, item in prepared.variants.items():
            response = await ctx.api.request(
                "POST",
                "/configurations/save",
                body={"deployable_configuration": item.configuration, "file": {"name": f"mcp-{slug(variant_id, 60)}"}},
                timeout=60,
            )
            saved[variant_id] = response["artifact"]["artifact_id"]
    except BaseException:
        await asyncio.shield(_discard(ctx, list(saved.values())))
        raise
    return saved


async def start(ctx: ToolContext, design: Any, expected_digest: str | None, client_request_id: str | None):
    name = client_name(design.name, client_request_id)
    if client_request_id and (found := await _existing(ctx, name)) is not None:
        return {"already_started": True, **_started(found, None, {})}
    prepared = await prepare(ctx, design)
    if expected_digest and expected_digest != prepared.digest:
        raise ToolError(
            "plan_stale",
            "The plan changed since plan_experiment (cluster, image or render output differ).",
            field="expected_digest",
            hint="Call plan_experiment again, review it, and start with its request_digest.",
            current_digest=prepared.digest,
        )
    artifacts = await _save_artifacts(ctx, prepared)
    body = evaluation_request(design, prepared.plans, artifacts, prepared.session_id, prepared.seed, name)
    try:
        workflow = await ctx.api.request("POST", "/evaluate/evaluations", body=body, timeout=120)
    except LensApiError as error:
        leftovers = await _discard(ctx, list(artifacts.values()))
        failure = upstream_error(error)
        message = failure.message
        if error.status == 409 and any(word in message.lower() for word in CAPACITY_WORDS):
            failure.code, failure.hint = "insufficient_capacity", "Free accelerators (see get_cluster) and start again."
        elif error.status == 409:
            failure.code = "conflict_state"
        failure.extra["leftover_artifacts"] = leftovers
        raise failure from error
    return _started(workflow, prepared, artifacts)


async def cancel(ctx: ToolContext, experiment_id: str, case_ids: list[str] | None) -> dict[str, Any]:
    workflow = await load_workflow(ctx, experiment_id)
    guard_owner(ctx, workflow)
    if workflow.get("status") in TERMINAL:
        return {"experiment_id": experiment_id, "status": workflow["status"], "outcome": "already_done"}
    known = {case.get("id"): case for case in workflow.get("cases") or []}
    unknown = sorted(set(case_ids or []) - set(known))
    if unknown:
        raise ToolError(
            "invalid_argument",
            f"Unknown case ids: {', '.join(unknown)}.",
            field="case_ids",
            hint="Use the case_id values shown by get_experiment(include=['cases']).",
        )
    paths = [f"{WORKFLOWS}/{experiment_id}/cases/{case_id}/cancel" for case_id in case_ids or []] or [
        f"{WORKFLOWS}/{experiment_id}/cancel"
    ]
    try:
        for path in paths:
            await ctx.api.request("POST", path, timeout=120)
    except LensApiError as error:
        if error.status == 409:
            raise ToolError(
                "cleanup_pending",
                error.message(),
                field="experiment_id",
                hint="Lens could not finish cleaning up; call cancel_experiment again.",
                retryable=True,
            ) from error
        raise
    after = await load_workflow(ctx, experiment_id)
    before = {case_id: case.get("status") for case_id, case in known.items()}
    cancelled = [
        case["id"]
        for case in after.get("cases") or []
        if case.get("status") in ("cancelled", "cancelling") and before.get(case["id"]) not in ("cancelled",)
    ]
    errors = [str(after[key])[:300] for key in ("cancellation_error",) if after.get(key)]
    settled = after.get("status") in TERMINAL or (case_ids and not errors)
    return {
        "experiment_id": experiment_id,
        "status": after.get("status"),
        "cancelled_case_ids": cancelled,
        "cleanup": {"state": "done" if settled and not errors else "pending", "errors": errors},
        "poll_after_seconds": 15,
    }


async def retry(ctx: ToolContext, experiment_id: str) -> dict[str, Any]:
    workflow = await load_workflow(ctx, experiment_id)
    guard_owner(ctx, workflow)
    if workflow.get("status") not in ("failed", "cancelled") or not workflow.get("cases"):
        raise ToolError(
            "conflict_state",
            f"Experiment is {workflow.get('status')}; only failed or cancelled experiments can be retried.",
            field="experiment_id",
            hint="Wait for it to finish, or cancel it first.",
        )
    requeued = [case["id"] for case in workflow["cases"] if case.get("status") in ("failed", "cancelled")]
    try:
        retried = await ctx.api.request("POST", f"{WORKFLOWS}/{experiment_id}/retry", timeout=60)
    except LensApiError as error:
        if error.status == 409:
            raise ToolError("conflict_state", error.message(), field="experiment_id") from error
        raise
    return {
        "experiment_id": experiment_id,
        "status": (retried or {}).get("status"),
        "requeued_case_ids": requeued,
        "poll_after_seconds": 30,
    }


def _mcp_artifacts(workflow: dict[str, Any]) -> list[str]:
    return sorted(
        {
            plan["configuration_artifact_id"]
            for plan in workflow.get("benchmark_plans") or []
            if str(plan.get("deployment_description") or "").startswith(MCP_PLAN_PREFIX)
            and plan.get("configuration_artifact_id")
        }
    )


def _preserved(workflow: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"kind": "deployment", "id": case["deployment_run_id"]}
        for case in workflow.get("cases") or []
        if case.get("preserve_deployment") and case.get("deployment_run_id")
    ]


async def expect_name(ctx: ToolContext, experiment_id: str, expected: str) -> None:
    """Refuses before any deletion when the name differs; unreadable experiments are left to delete()."""
    try:
        name = (await load_workflow(ctx, experiment_id)).get("name")
    except (ToolError, LensApiError, TimeoutError):
        return
    if name != expected:
        raise ToolError(
            "id_mismatch",
            f"{experiment_id} is named '{name}', not '{expected}'. Nothing was deleted.",
            field="expected_names",
            hint="Re-run list_resources and pass the ids and names you actually mean.",
        )


async def delete(ctx: ToolContext, experiment_id: str) -> dict[str, Any]:
    """Outcome of deleting one Lens experiment: cascades to its deployment runs, history and saved artifacts."""
    if not is_path_id(experiment_id):
        return {
            "id": experiment_id,
            "kind": "experiment",
            "status": "failed",
            "error": "invalid_argument",
            "message": "Not a valid experiment id.",
        }
    try:
        workflow = await load_workflow(ctx, experiment_id)
    except ToolError:
        return {"id": experiment_id, "kind": "experiment", "status": "not_found", "error": "not_found"}
    except LensApiError as error:
        return {
            "id": experiment_id,
            "kind": "experiment",
            "status": "failed",
            "error": error.code() or f"lens_{error.status}",
            "message": error.message(),
        }
    except TimeoutError:
        return {
            "id": experiment_id,
            "kind": "experiment",
            "status": "failed",
            "error": "timeout",
            "message": "Lens did not answer in time.",
        }
    try:
        guard_owner(ctx, workflow)
    except ToolError as error:
        return {
            "id": experiment_id,
            "kind": "experiment",
            "status": "failed",
            "error": error.code,
            "message": error.message,
        }
    if workflow.get("status") not in TERMINAL:
        return {
            "id": experiment_id,
            "kind": "experiment",
            "status": "blocked",
            "error": "active",
            "message": f"The experiment is {workflow.get('status')}; cancel_experiment it first.",
        }
    also_deletes = _preserved(workflow)
    try:
        await ctx.api.request("DELETE", f"{WORKFLOWS}/{experiment_id}", timeout=300)
    except LensApiError as error:
        status = "blocked" if error.status == 409 else "failed"
        return {
            "id": experiment_id,
            "kind": "experiment",
            "status": status,
            "error": error.code(),
            "message": error.message(),
        }
    artifacts = _mcp_artifacts(workflow)
    leftovers = await _discard(ctx, artifacts)
    result: dict[str, Any] = {
        "id": experiment_id,
        "kind": "experiment",
        "status": "deleted",
        "name": workflow.get("name"),
        "also_deleted": also_deletes,
        "artifacts_deleted": [item for item in artifacts if item not in leftovers],
    }
    if leftovers:
        result["artifacts_left"] = leftovers
    return result


async def list_experiments(ctx: ToolContext, cluster_id: str | None, owner: str) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", WORKFLOWS, timeout=60)
    entries = []
    for item in sorted(data.get("items") or [], key=lambda w: w.get("created_at") or "", reverse=True):
        owned = item.get("owner_user_id") == ctx.owner or (item.get("owner_user_id") is None and ctx.owner == "system")
        if (owner == "me" and not owned) or (cluster_id and item.get("cluster_id") != cluster_id):
            continue
        entries.append(
            {
                "id": item["id"],
                "kind": "experiment",
                "name": item.get("name"),
                "ownership": "owned" if owned else "shared" if item.get("owner_user_id") else "unknown",
                "state": item.get("status"),
                "cluster_id": item.get("cluster_id"),
                "created_at": item.get("created_at"),
            }
        )
    return entries[:LIST_LIMIT]
