"""Writes on Lens Evaluation workflows: start, cancel, retry, delete and listing. No local state is kept."""

from __future__ import annotations

import asyncio
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .evaluation_design import evaluation_request
from .evaluation_plan import MAX_LISTED_CASES, Prepared, eta, prepare
from .kinds import owner_state
from .lens_api import LensApiError
from .path_ids import check_id
from .plan_review import EXPERIMENT_REVIEW, plan_rejection, review_view
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
                hint="Call resource_list(kinds=['experiment']) for ids.",
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
            hint="You can read it with experiment_get but not change it.",
        )


def review_required(workflow: dict[str, Any]) -> dict[str, Any]:
    """The plans Lens froze for an experiment that waits for confirmation (prefill/decode deployments)."""
    return {
        "plans": [
            {"case_id": item.get("case_id"), **review_view(item)} for item in workflow.get("prepared_plans") or []
        ],
        "next": EXPERIMENT_REVIEW,
    }


def _started(workflow: dict[str, Any], prepared: Prepared, artifact_ids: dict[str, str]) -> dict[str, Any]:
    cases = workflow.get("cases") or []
    awaiting = workflow.get("status") == "awaiting_review"
    return {
        "experiment_id": workflow["id"],
        "status": workflow.get("status"),
        "cases_total": len(cases),
        "artifact_ids": artifact_ids,
        **({"review_required": review_required(workflow)} if awaiting else {"poll_after_seconds": 60}),
        "next": (
            f"Not started yet: {EXPERIMENT_REVIEW}"
            if awaiting
            else f"Call experiment_get(experiment_id='{workflow['id']}', wait_seconds=30) to follow it."
        ),
        "case_order": prepared.cases[:MAX_LISTED_CASES],
        "seed": prepared.seed,
        "eta_seconds": eta(prepared),
    }


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


async def start(ctx: ToolContext, design: Any, expected_digest: str | None):
    prepared = await prepare(ctx, design)
    if expected_digest and expected_digest != prepared.digest:
        raise ToolError(
            "plan_stale",
            "The plan changed since experiment_plan (cluster, image or render output differ).",
            field="expected_digest",
            hint="Call experiment_plan again, review it, and start with its request_digest.",
            current_digest=prepared.digest,
        )
    artifacts = await _save_artifacts(ctx, prepared)
    body = evaluation_request(design, prepared.plans, artifacts, prepared.session_id, prepared.seed, design.name)
    try:
        workflow = await ctx.api.request("POST", "/evaluate/evaluations", body=body, timeout=120)
    except LensApiError as error:
        leftovers = await _discard(ctx, list(artifacts.values()))
        failure = upstream_error(error)
        message = failure.message
        if error.status == 409 and any(word in message.lower() for word in CAPACITY_WORDS):
            failure.code, failure.hint = "insufficient_capacity", "Free accelerators (see cluster_get) and start again."
        elif error.status == 409:
            failure.code = "conflict_state"
        failure.extra["leftover_artifacts"] = leftovers
        raise failure from error
    return _started(workflow, prepared, artifacts)


async def confirm(
    ctx: ToolContext, experiment_id: str, plan_digests: dict[str, str], acknowledge_unknown: bool
) -> dict[str, Any]:
    """Confirm the reviewed plans of an experiment that is waiting for it; Lens then queues it."""
    workflow = await load_workflow(ctx, experiment_id)
    guard_owner(ctx, workflow)
    if workflow.get("status") != "awaiting_review":
        raise ToolError(
            "conflict_state",
            f"The experiment is {workflow.get('status')}, not awaiting_review.",
            field="experiment_id",
            hint="Only an experiment that experiment_get reports as awaiting_review can be confirmed.",
        )
    try:
        confirmed = await ctx.api.request(
            "POST",
            f"/evaluate/evaluations/{experiment_id}/confirm",
            body={"plan_digests": plan_digests, "acknowledge_unknown": acknowledge_unknown},
            timeout=60,
        )
    except LensApiError as error:
        raise plan_rejection(error) from error
    return {
        "experiment_id": experiment_id,
        "status": confirmed.get("status"),
        "poll_after_seconds": 60,
        "next": f"Call experiment_get(experiment_id='{experiment_id}', wait_seconds=30) to follow it.",
    }


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
            hint="Use the case_id values shown by experiment_get(include=['cases']).",
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
                hint="Lens could not finish cleaning up; call experiment_cancel again.",
                retryable=True,
            ) from error
        raise
    except TimeoutError as error:
        raise ToolError(
            "cleanup_pending",
            "Lens is still stopping the experiment and cleaning up its deployments.",
            field="experiment_id",
            hint="Call experiment_get to watch it, or experiment_cancel again once it has settled.",
            retryable=True,
        ) from error
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
    retried = retried or {}
    if retried.get("status") == "awaiting_review":
        # Cases that need a new deployment get fresh plans, which must be reviewed again.
        return {
            "experiment_id": experiment_id,
            "status": "awaiting_review",
            "requeued_case_ids": requeued,
            "review_required": review_required(retried),
            "next": f"Not restarted yet: {EXPERIMENT_REVIEW}",
        }
    return {
        "experiment_id": experiment_id,
        "status": retried.get("status"),
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


async def get_entry(ctx: ToolContext, experiment_id: str) -> dict[str, Any] | None:
    try:
        workflow = await load_workflow(ctx, experiment_id)
    except ToolError:
        return None
    return _entry(ctx, workflow)


def _entry(ctx: ToolContext, workflow: dict[str, Any]) -> dict[str, Any]:
    owner = workflow.get("owner_user_id")
    return {
        "id": workflow["id"],
        "kind": "experiment",
        "name": workflow.get("name"),
        "status": workflow.get("status"),
        "ownership": owner_state(ctx, owner),
        "cluster_id": workflow.get("cluster_id"),
        "created_at": workflow.get("created_at"),
    }


async def delete(ctx: ToolContext, experiment_id: str) -> list[dict[str, str]]:
    """Delete a finished experiment; it takes its preserved deployment runs, history and saved artifacts along."""
    workflow = await load_workflow(ctx, experiment_id)
    guard_owner(ctx, workflow)
    if workflow.get("status") not in TERMINAL:
        raise ToolError(
            "active",
            f"The experiment is {workflow.get('status')}; experiment_cancel it first.",
            field="experiment_id",
        )
    await ctx.api.request("DELETE", f"{WORKFLOWS}/{experiment_id}", timeout=300)
    artifacts = _mcp_artifacts(workflow)
    leftovers = await _discard(ctx, artifacts)
    return [
        *_preserved(workflow),
        *({"kind": "configuration_artifact", "id": item} for item in artifacts if item not in leftovers),
    ]


async def list_experiments(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    data = await ctx.api.request("GET", WORKFLOWS, timeout=60)
    items = sorted(data.get("items") or [], key=lambda w: w.get("created_at") or "", reverse=True)
    return [_entry(ctx, item) for item in items if not cluster_id or item.get("cluster_id") == cluster_id][:LIST_LIMIT]
