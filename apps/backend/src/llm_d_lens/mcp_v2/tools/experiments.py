"""Experiments on Lens Evaluation workflows; legacy ``exp-`` ids are routed to the frozen legacy path."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .. import evaluation_ops as ops
from .. import experiments_legacy as legacy
from ..context import ToolContext
from ..errors import ToolError
from ..evaluation_design import ExperimentDesign
from ..evaluation_plan import describe, prepare
from ..evaluation_results import case_rows, compare, result_view, run_row
from ..evaluation_view import case_view, poll_after, remaining_seconds, summarize, watch
from ..experiments_engine import delete_experiment, dependents_of_experiment, describe_experiment
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..path_ids import PathId, check_id
from ..registry import tool

Include = Literal["cases", "results", "comparison", "points", "timeline"]
Metric = Literal[
    "ttft_ms.p50",
    "ttft_ms.p90",
    "ttft_ms.p99",
    "tpot_ms.p50",
    "tpot_ms.p90",
    "tpot_ms.p99",
    "itl_ms.p50",
    "itl_ms.p99",
    "e2e_latency_ms.p50",
    "e2e_latency_ms.p99",
    "output_tokens_per_s",
    "requests_per_s",
    "success_rate",
]
EXPERIMENT_ID = {"min_length": 1, "max_length": 100, "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"}
MAX_WAIT_SECONDS = 50


class PlanExperimentInput(ExperimentDesign):
    """The experiment design."""


class StartExperimentInput(ExperimentDesign):
    expected_digest: str | None = Field(
        None, description="request_digest from plan_experiment; start fails with plan_stale if the plan changed."
    )
    client_request_id: str | None = Field(
        None, max_length=64, description="Retry key: repeating it returns the first result."
    )


class GetExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(**EXPERIMENT_ID, description="Experiment id returned by start_experiment.")
    include: list[Include] = Field(
        default_factory=list,
        description="cases=every case; results=measurements per plan and scenario; comparison=deltas versus the "
        "first plan; points=every concurrency/rate point; timeline=case start and finish times.",
    )
    wait_seconds: int = Field(0, ge=0, le=MAX_WAIT_SECONDS, description="Really wait up to this many seconds.")
    wait_for: Literal["change", "terminal"] = Field(
        "change", description="change=return when progress changes; terminal=return only when it has finished."
    )


class CancelExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(**EXPERIMENT_ID, description="Experiment to cancel.")
    case_ids: list[str] | None = Field(
        None, min_length=1, max_length=50, description="Cancel only these cases; default is the whole experiment."
    )
    reason: str | None = Field(None, max_length=200, description="Why it was cancelled (informational).")


class RetryExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(**EXPERIMENT_ID, description="A failed or cancelled experiment.")


class CompareResultsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[PathId] = Field(default_factory=list, max_length=10, description="Lens experiment ids.")
    benchmark_run_ids: list[PathId] = Field(default_factory=list, max_length=50, description="Standalone Lens runs.")
    baseline: str | None = Field(None, description="Plan label that deltas are computed against; default the first.")
    metrics: list[Metric] | None = Field(None, description="Default: ttft p99, tpot p99, throughput, success rate.")
    group_repetitions: bool = Field(True, description="Report the median of repetitions instead of one row each.")
    result_refs: list[str] | None = Field(
        None, max_length=20, description="Legacy exp- result references; cannot be combined with the fields above."
    )
    baseline_ref: str | None = Field(None, description="Legacy: reference deltas are computed against.")
    normalize: Literal["none", "per_gpu"] = Field("none", description="Legacy: per_gpu divides throughput by gpus.")


@tool(
    "plan_experiment",
    "[R] Plan an experiment on Lens Evaluation workflows without running anything: every variant is planned with the "
    "guide planner and rendered, then scenarios x variants x repetitions are expanded into cases. Returns the plans, "
    "the case order, peak accelerators, an ETA, warnings, the shuffle seed and a request_digest to pass to "
    "start_experiment as expected_digest. Use plan_deployment for a single deployment.\n"
    "Side effects: none; resources: none; duration: synchronous, up to ~2 min for many variants; reversible: n/a.",
    PlanExperimentInput,
    read_only=True,
    open_world=True,
)
async def plan_experiment(ctx: ToolContext, args: PlanExperimentInput) -> dict[str, Any]:
    prepared = await prepare(ctx, args)
    result = describe(prepared)
    result["status"] = "valid_with_warnings" if result["warnings"] else "valid"
    result["next"] = "Call start_experiment with the same design and expected_digest=request_digest."
    return result


@tool(
    "start_experiment",
    "[S] Start an experiment as one Lens Evaluation workflow and return immediately; Lens runs it server-side: it "
    "deploys each variant, runs the scenarios (and optional baselines), and deletes the deployments unless "
    "preserve_deployment is set. This tool saves one configuration artifact per variant, creates the workflow "
    "(experiment id = Lens workflow id) and never waits or advances anything; follow it with get_experiment. "
    "Pass expected_digest from plan_experiment to refuse a changed plan. Stop it with cancel_experiment, rerun a "
    "failed one with retry_experiment, remove it with delete_resources.\n"
    "Side effects: creates configuration artifacts and a Lens evaluation workflow that deploys and benchmarks; "
    "resources: accelerators while it runs; duration: returns in seconds, the experiment takes minutes to hours; "
    "reversible: cancel_experiment.",
    StartExperimentInput,
    read_only=False,
    open_world=True,
)
async def start_experiment(ctx: ToolContext, args: StartExperimentInput) -> dict[str, Any]:
    design = ExperimentDesign(**{name: getattr(args, name) for name in ExperimentDesign.model_fields})
    return await ops.start(ctx, design, args.expected_digest, args.client_request_id)


def _not_found(experiment_id: str) -> ToolError:
    return ToolError(
        "not_found",
        f"No experiment '{experiment_id}'.",
        field="experiment_id",
        hint="Call list_resources(kinds=['experiment']) for ids.",
    )


def _timeline(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    events = [("created", workflow.get("created_at"), None), ("started", workflow.get("started_at"), None)]
    for case in workflow.get("cases") or []:
        events.append(("case_started", case.get("started_at"), case.get("id")))
        events.append((f"case_{case.get('status')}", case.get("finished_at"), case.get("id")))
    events.append((str(workflow.get("status")), workflow.get("finished_at"), None))
    items = [{"at": at, "event": name, **({"case_id": cid} if cid else {})} for name, at, cid in events if at]
    return sorted(items, key=lambda item: item["at"])


async def _details(ctx: ToolContext, experiment_id: str) -> dict[str, Any]:
    check_id(experiment_id, "experiment_id")
    try:
        return await ctx.api.request("GET", f"/evaluate/workflow-runs/{experiment_id}/details", timeout=120)
    except LensApiError as error:
        if error.status in (404, 422):
            raise _not_found(experiment_id) from error
        raise


def _poll_phase(view: dict[str, Any]) -> str:
    if view.get("status") in ops.TERMINAL:
        return "terminal"
    if view.get("status") == "cancelling":
        return "cancelling"
    return (view.get("active_case") or {}).get("status") or "queued"


@tool(
    "get_experiment",
    "[R] Status and results of an experiment. Returns status, progress (cases by state), the active case with its "
    "phase and heartbeat, last_progress_at, a stale verdict with reason, a heuristic ETA and poll_after_seconds. "
    "wait_seconds (max 50) really waits: wait_for='change' returns when progress changes, 'terminal' only when it "
    "finishes; the reply says whether it timed out. include=['results'] gives measurements per plan, scenario and "
    "repetition; 'comparison' adds deltas versus the first plan; 'points' adds every sweep point; 'cases' lists "
    "every case. Legacy exp- ids are read through the old path. Never changes the experiment.\n"
    "Side effects: none; resources: none; duration: up to wait_seconds; reversible: n/a.",
    GetExperimentInput,
    read_only=True,
)
async def get_experiment(ctx: ToolContext, args: GetExperimentInput) -> dict[str, Any]:
    if legacy.is_legacy(args.experiment_id):
        return await legacy.get(ctx, args.experiment_id, list(args.include), args.wait_seconds, args.wait_for)
    try:
        workflow, active, waited, outcome = await watch(ctx, args.experiment_id, args.wait_seconds, args.wait_for)
    except LensApiError as error:
        if error.status in (404, 422):
            raise _not_found(args.experiment_id) from error
        raise
    view = summarize(ctx, workflow, active)
    eta_seconds = await remaining_seconds(ctx, workflow)
    if eta_seconds is not None:
        view["eta_remaining_seconds"] = {"value": eta_seconds, "basis": "mcp-heuristic"}
    view["poll_after_seconds"] = poll_after(_poll_phase(view), eta_seconds)
    if args.wait_seconds:
        view.update(waited_seconds=waited, wait_outcome=outcome)
    include = set(args.include)
    if "cases" in include:
        view["cases"] = [case_view(case) for case in workflow.get("cases") or []]
    if include & {"results", "comparison", "points"}:
        rows = case_rows(await _details(ctx, args.experiment_id), args.experiment_id)
        view["results"] = result_view(rows, "points" in include)
        if "comparison" in include:
            view["comparison"] = compare(rows, None, None, True)
    if "timeline" in include:
        view["timeline"] = _timeline(workflow)
    return view


@tool(
    "cancel_experiment",
    "[D] Cancel an experiment through Lens: the running benchmark is stopped and Lens deletes the deployments it "
    "created; finished measurements are kept. case_ids cancels only those cases. Returns the status, the cancelled "
    "case ids and the cleanup state (cleanup_pending means call it again). Cancelling a finished experiment is "
    "harmless. Legacy exp- ids are cancelled through the old path. Only the creator can cancel.\n"
    "Side effects: aborts running benchmarks and deletes the experiment's deployments; resources: frees their "
    "accelerators; duration: up to ~2 min while Lens cleans up; reversible: no (retry_experiment reruns).",
    CancelExperimentInput,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def cancel_experiment(ctx: ToolContext, args: CancelExperimentInput) -> dict[str, Any]:
    if legacy.is_legacy(args.experiment_id):
        return await legacy.cancel(ctx, args.experiment_id, args.reason)
    return await ops.cancel(ctx, args.experiment_id, args.case_ids)


@tool(
    "retry_experiment",
    "[S] Retry a failed or cancelled experiment: Lens requeues its unfinished cases and runs them again with the "
    "same configuration (finished cases keep their results). Follow it with get_experiment. Only the creator can "
    "retry; other states return conflict_state.\n"
    "Side effects: requeues cases and redeploys; resources: accelerators while it runs; duration: returns in "
    "seconds; reversible: cancel_experiment.",
    RetryExperimentInput,
    read_only=False,
    open_world=True,
)
async def retry_experiment(ctx: ToolContext, args: RetryExperimentInput) -> dict[str, Any]:
    if legacy.is_legacy(args.experiment_id):
        raise ToolError(
            "conflict_state",
            "Legacy exp- experiments cannot be retried.",
            field="experiment_id",
            hint="Start a new experiment with start_experiment.",
        )
    return await ops.retry(ctx, args.experiment_id)


@tool(
    "compare_results",
    "[R] Compare measurements across Lens experiments and standalone benchmark runs: median, min, max and the delta "
    "versus a baseline plan per scenario and concurrency or rate point, with SLA verdicts. Repetitions are "
    "grouped by default. Legacy exp- result_refs are compared through the old path.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    CompareResultsInput,
    read_only=True,
)
async def compare_results(ctx: ToolContext, args: CompareResultsInput) -> dict[str, Any]:
    if args.result_refs:
        if args.experiment_ids or args.benchmark_run_ids:
            raise ToolError(
                "invalid_argument",
                "result_refs cannot be combined with experiment_ids or benchmark_run_ids.",
                field="result_refs",
            )
        return legacy.compare(ctx, args.result_refs, args.baseline_ref, args.metrics, args.normalize)
    if not args.experiment_ids and not args.benchmark_run_ids:
        raise ToolError(
            "invalid_argument",
            "Pass experiment_ids or benchmark_run_ids.",
            field="experiment_ids",
            hint="Use ids from start_experiment or list_resources(kinds=['experiment']).",
        )
    rows: list[dict[str, Any]] = []
    for experiment_id in dict.fromkeys(args.experiment_ids):
        if legacy.is_legacy(experiment_id):
            raise ToolError(
                "invalid_argument",
                f"'{experiment_id}' is a legacy experiment; pass its result_refs instead.",
                field="experiment_ids",
            )
        rows += case_rows(await _details(ctx, experiment_id), experiment_id)
    for run_id in dict.fromkeys(args.benchmark_run_ids):
        try:
            run = await ctx.api.request("GET", f"/evaluate/runs/{run_id}", timeout=60)
        except LensApiError as error:
            if error.status in (404, 422):
                raise ToolError("not_found", f"No benchmark run '{run_id}'.", field="benchmark_run_ids") from error
            raise
        rows.append({**run_row(run), "source": run_id})
    comparison = compare(rows, args.baseline, list(args.metrics or []) or None, args.group_repetitions)
    if comparison.get("error"):
        raise ToolError(
            "invalid_argument",
            f"Baseline '{args.baseline}' is not among the compared plans.",
            field="baseline",
            hint=f"Plans: {', '.join(comparison['plans'])}.",
        )
    return {"comparison": comparison, "results": result_view(rows, False)}


register_kind("experiment", delete=delete_experiment, describe=describe_experiment, dependents=dependents_of_experiment)
