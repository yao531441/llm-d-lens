"""``get_experiment``: a compact, derived view of a Lens Evaluation workflow and a bounded real wait (design 7.5)."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from .context import ToolContext
from .evaluation_design import canonical
from .lens_api import LensApiError
from .path_ids import check_id, is_path_id

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
ACTIVE_CASE = frozenset({"deploying", "benchmarking", "cancelling"})
POLL_SECONDS = 5
DEPLOY_ESTIMATE_SECONDS = 120
QUEUED_STALE_SECONDS = 120
DEPLOY_STALE_SECONDS = 2 * 3 * DEPLOY_ESTIMATE_SECONDS
HEARTBEAT_STALE_SECONDS = 60
CANCEL_STALE_SECONDS = 300
sleep = asyncio.sleep


def now() -> datetime:
    return datetime.now(UTC)


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _latest(*candidates: tuple[str, Any]) -> tuple[str | None, str | None]:
    """The most recent of (source field, ISO time) pairs, as ``(time, source)``."""
    best: tuple[datetime, str, str] | None = None
    for source, value in candidates:
        moment = parse_time(value)
        if moment and (best is None or moment > best[0]):
            best = (moment, source, value)
    return (best[2], best[1]) if best else (None, None)


async def fetch_workflow(ctx: ToolContext, experiment_id: str) -> dict[str, Any]:
    check_id(experiment_id, "experiment_id")
    return await ctx.api.request("GET", f"/evaluate/workflow-runs/{experiment_id}", timeout=30)


def active_case(workflow: dict[str, Any]) -> dict[str, Any] | None:
    cases = workflow.get("cases") or []
    active_id = workflow.get("active_case_id")
    chosen = next((case for case in cases if case.get("id") == active_id), None) if active_id else None
    return chosen or next((case for case in cases if case.get("status") in ACTIVE_CASE), None)


async def fetch_active(ctx: ToolContext, case: dict[str, Any] | None) -> dict[str, Any] | None:
    """The benchmark run (or deployment run) behind the active case; a failed read must not fail the summary."""
    if case is None:
        return None
    if case.get("status") == "benchmarking" and is_path_id(case.get("evaluation_run_id")):
        path = f"/evaluate/runs/{case['evaluation_run_id']}"
    elif case.get("status") == "deploying" and is_path_id(case.get("deployment_run_id")):
        path = f"/deployments/runs/{case['deployment_run_id']}"
    else:
        return None
    try:
        return await ctx.api.request("GET", path, timeout=30)
    except (LensApiError, TimeoutError):
        return None


def progress(workflow: dict[str, Any]) -> dict[str, Any]:
    cases = workflow.get("cases") or []
    counts = {status: sum(1 for case in cases if case.get("status") == status) for status in TERMINAL}
    done = sum(counts.values())
    return {
        "cases_total": len(cases),
        **counts,
        "active": sum(1 for case in cases if case.get("status") in ACTIVE_CASE),
        "queued": sum(1 for case in cases if case.get("status") == "queued"),
        "fraction": round(done / len(cases), 3) if cases else 0.0,
    }


def phase_of(workflow: dict[str, Any], case: dict[str, Any] | None) -> str:
    status = workflow.get("status")
    if status in TERMINAL:
        return "terminal"
    if status == "cancelling":
        return "cancelling"
    if case and case.get("status") in ("deploying", "benchmarking"):
        return case["status"]
    return "queued"


def _age(moment: str | None) -> float | None:
    parsed = parse_time(moment)
    return None if parsed is None else (now() - parsed).total_seconds()


def progress_marker(
    workflow: dict[str, Any], case: dict[str, Any] | None, active: dict[str, Any] | None, phase: str
) -> tuple[str | None, str | None]:
    """``(time, source field)`` of the latest sign of life; Lens workflows carry no heartbeat of their own."""
    cases = workflow.get("cases") or []
    if phase == "terminal":
        return workflow.get("finished_at"), "finished_at"
    if phase == "benchmarking" and active:
        # Lens sets heartbeat_at only once metrics are awaited; earlier preparation phases stamp phase_started_at.
        return _latest(
            ("evaluation_run.heartbeat_at", active.get("heartbeat_at")),
            ("evaluation_run.phase_started_at", active.get("phase_started_at")),
            ("evaluation_run.started_at", active.get("started_at")),
        )
    if phase == "deploying" and case:
        return _latest(
            ("case.started_at", case.get("started_at")),
            ("deployment_run.started_at", (active or {}).get("started_at")),
        )
    return _latest(
        ("created_at", workflow.get("created_at")),
        ("started_at", workflow.get("started_at")),
        *(("case.finished_at", item.get("finished_at")) for item in cases),
        *(("case.started_at", item.get("started_at")) for item in cases),
    )


def stale_check(
    workflow: dict[str, Any], phase: str, marker: str | None, case: dict[str, Any] | None
) -> dict[str, Any]:
    age = _age(marker)
    verdicts = {
        "queued": (
            case is None and age is not None and age > QUEUED_STALE_SECONDS,
            "No case has started and Lens has shown no activity.",
            "Lens has not started the workflow; queued workflows are resumed on Lens startup.",
        ),
        "deploying": (
            age is not None and age > DEPLOY_STALE_SECONDS,
            f"The deployment has not become ready for more than {DEPLOY_STALE_SECONDS} seconds.",
            "Call diagnose_deployment on the case's deployment run.",
        ),
        "benchmarking": (
            age is not None and age > HEARTBEAT_STALE_SECONDS,
            f"The benchmark run has not reported for more than {HEARTBEAT_STALE_SECONDS} seconds.",
            "The benchmark runner stopped reporting; after a Lens restart the run is marked failed and the case "
            "is rerun from its start.",
        ),
        "cancelling": (
            bool(workflow.get("cancellation_error")) or (age is not None and age > CANCEL_STALE_SECONDS),
            "Cancellation cleanup has not finished.",
            "Cleanup did not finish; call cancel_experiment again.",
        ),
    }
    is_stale, reason, hint = verdicts.get(phase, (False, "", ""))
    result: dict[str, Any] = {"is_stale": bool(is_stale)}
    if is_stale:
        result.update(reason=reason, hint=hint)
    return result


def poll_after(phase: str, remaining_seconds: int | None) -> int | None:
    if phase == "terminal":
        return None
    if phase == "cancelling":
        return 15
    if phase == "benchmarking":
        return max(15, min(60, (remaining_seconds if remaining_seconds is not None else 240) // 4))
    return 30


def case_view(case: dict[str, Any]) -> dict[str, Any]:
    view = {
        "case_id": case.get("id"),
        "kind": case.get("kind"),
        "status": case.get("status"),
        "plan": case.get("benchmark_plan_id"),
        "scenario": case.get("scenario_id"),
        "baseline_type": case.get("baseline_type"),
        "started_at": case.get("started_at"),
        "finished_at": case.get("finished_at"),
        "run_id": case.get("evaluation_run_id"),
        "deployment_run_id": case.get("deployment_run_id"),
        "error": str(case["error"])[:300] if case.get("error") else None,
    }
    return {key: value for key, value in view.items() if value is not None}


def fingerprint(workflow: dict[str, Any], active: dict[str, Any] | None) -> str:
    case = active_case(workflow)
    return canonical(
        {
            "status": workflow.get("status"),
            "progress": progress(workflow),
            "active": [case.get("id"), case.get("status")] if case else None,
            "phase": (active or {}).get("phase"),
        }
    )


async def remaining_seconds(ctx: ToolContext, workflow: dict[str, Any]) -> int | None:
    """Lens timing estimate of the unfinished cases plus a deploy heuristic; ``None`` when nothing is left."""
    pending = [c for c in workflow.get("cases") or [] if c.get("status") in ("queued", "deploying", "benchmarking")]
    if not pending:
        return None
    unique = {canonical(c["benchmark"]): c["benchmark"] for c in pending if isinstance(c.get("benchmark"), dict)}
    estimates: dict[str, int] = {}
    for key, benchmark in list(unique.items())[:20]:
        try:
            estimate = await ctx.api.request("POST", "/evaluate/timing-estimate", body=benchmark, timeout=30)
        except (LensApiError, TimeoutError):
            continue
        estimates[key] = (int(estimate.get("lower_seconds") or 0) + int(estimate.get("upper_seconds") or 0)) // 2
    benchmark_seconds = sum(estimates.get(canonical(c.get("benchmark")), 0) for c in pending)
    groups = {c.get("deployment_group_id") for c in pending if not c.get("deployment_run_id")}
    deploys = len([g for g in groups if g and "kubernetes-service" not in str(g)])
    return benchmark_seconds + deploys * DEPLOY_ESTIMATE_SECONDS


def summarize(ctx: ToolContext, workflow: dict[str, Any], active: dict[str, Any] | None) -> dict[str, Any]:
    case = active_case(workflow)
    phase = phase_of(workflow, case)
    marker, source = progress_marker(workflow, case, active, phase)
    summary: dict[str, Any] = {
        "experiment_id": workflow.get("id"),
        "name": workflow.get("name"),
        "status": workflow.get("status"),
        "cluster_id": workflow.get("cluster_id"),
        "owned_by_me": None if not workflow.get("owner_user_id") else workflow["owner_user_id"] == ctx.owner,
        "created_at": workflow.get("created_at"),
        "started_at": workflow.get("started_at"),
        "finished_at": workflow.get("finished_at"),
        "progress": progress(workflow),
        "last_progress_at": {"value": marker, "source": source},
        "stale": stale_check(workflow, phase, marker, case),
    }
    if case:
        view = case_view(case)
        if active:
            view["phase"] = active.get("phase")
            view["heartbeat_at"] = active.get("heartbeat_at")
        summary["active_case"] = {key: value for key, value in view.items() if value is not None}
    for key in ("error", "cancellation_error"):
        if workflow.get(key):
            summary[key] = str(workflow[key])[:400]
    return summary


async def watch(
    ctx: ToolContext, experiment_id: str, wait_seconds: int, wait_for: str
) -> tuple[dict[str, Any], dict[str, Any] | None, int, str | None]:
    """Poll Lens every 5 seconds until the condition holds or ``wait_seconds`` elapse; returns the latest state."""
    started = time.monotonic()
    slept = 0.0
    workflow = await fetch_workflow(ctx, experiment_id)
    active = await fetch_active(ctx, active_case(workflow))
    first = fingerprint(workflow, active)
    outcome = None
    while wait_seconds:
        if workflow.get("status") in TERMINAL:
            outcome = "terminal"
            break
        left = wait_seconds - max(slept, time.monotonic() - started)
        if left <= 0:
            outcome = "timeout"
            break
        step = min(POLL_SECONDS, left)
        await sleep(step)
        slept += step
        workflow = await fetch_workflow(ctx, experiment_id)
        active = await fetch_active(ctx, active_case(workflow))
        if wait_for == "change" and fingerprint(workflow, active) != first:
            outcome = "terminal" if workflow.get("status") in TERMINAL else "changed"
            break
    waited = int(round(max(slept, time.monotonic() - started))) if wait_seconds else 0
    return workflow, active, waited, outcome
