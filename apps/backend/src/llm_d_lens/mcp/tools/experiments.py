"""Experiments on Lens Evaluation workflows."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .. import evaluation_ops as ops
from ..context import ToolContext
from ..errors import ToolError
from ..evaluation_design import ExperimentDesign
from ..evaluation_plan import describe, prepare
from ..evaluation_results import case_rows, compare, result_view, run_row
from ..evaluation_view import case_view, poll_after, remaining_seconds, summarize, watch
from ..guide_fields import PD_GUIDE
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..path_ids import PathId, check_id
from ..planning import NOT_CHECKED, invalid_plan, location_caveat
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
MAX_ARTIFACT_CHARS = 4000


class PlanExperimentInput(ExperimentDesign):
    """The experiment design."""


class StartExperimentInput(ExperimentDesign):
    expected_digest: str | None = Field(
        None, description="request_digest from experiment_plan; start fails with plan_stale if the plan changed."
    )


class GetExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(**EXPERIMENT_ID, description="Experiment id returned by experiment_start.")
    include: list[Include] = Field(
        default_factory=list,
        description="cases=every case; results=measurements per plan and scenario; comparison=deltas versus the "
        "first plan; points=every concurrency/rate point; timeline=case start and finish times.",
    )
    wait_seconds: int = Field(0, ge=0, le=MAX_WAIT_SECONDS, description="Really wait up to this many seconds.")
    wait_for: Literal["change", "terminal"] = Field(
        "change", description="change=return when progress changes; terminal=return only when it has finished."
    )


class ConfirmExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(**EXPERIMENT_ID, description="An experiment that is awaiting_review.")
    plan_digests: dict[str, str] = Field(
        min_length=1,
        max_length=64,
        description="plan_id -> plan_digest for every plan in review_required.plans, copied exactly.",
    )
    acknowledge_unknown: bool = Field(
        False,
        description="Accept the scheduling constraints preflight could not check (plans[].preflight."
        "unknown_constraints). Only after the user has seen them.",
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


class GetArtifactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(**EXPERIMENT_ID, description="Experiment id returned by experiment_start.")
    case_id: str = Field(min_length=1, max_length=100, description="Case id from experiment_get(include=['cases']).")
    name: str | None = Field(
        None,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_./-]*$",
        description="Artifact name from the list, for example per_request_lifecycle_metrics.json. Omit to list.",
    )


class CompareResultsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[PathId] = Field(default_factory=list, max_length=10, description="Lens experiment ids.")
    benchmark_run_ids: list[PathId] = Field(default_factory=list, max_length=50, description="Standalone Lens runs.")
    baseline: str | None = Field(None, description="Plan label that deltas are computed against; default the first.")
    metrics: list[Metric] | None = Field(None, description="Default: ttft p99, tpot p99, throughput, success rate.")
    group_repetitions: bool = Field(True, description="Report the median of repetitions instead of one row each.")


@tool(
    "experiment_plan",
    "[R] Plan an experiment on Lens Evaluation workflows without running anything: every variant is planned with the "
    "guide planner and rendered, then scenarios x variants x repetitions are expanded into cases. Returns the plans, "
    "the case order, peak accelerators, an ETA, warnings, the shuffle seed and a request_digest to pass to "
    "experiment_start as expected_digest. Use deployment_plan for a single deployment.\n"
    "Side effects: none; resources: none; duration: synchronous, up to ~2 min for many variants; reversible: n/a.",
    PlanExperimentInput,
    read_only=True,
    open_world=True,
)
async def plan_experiment(ctx: ToolContext, args: PlanExperimentInput) -> dict[str, Any]:
    try:
        prepared = await prepare(ctx, args)
    except ToolError as error:
        return invalid_plan(error, "Fix the error and call experiment_plan again.")
    result = describe(prepared)
    result["status"] = "valid_with_warnings" if result["warnings"] else "valid"
    result["not_checked"] = NOT_CHECKED
    review = (
        " It deploys prefill/decode sets, so Lens will hold it as awaiting_review until the user has reviewed the "
        "plans and you call experiment_confirm."
        if any(variant.fields["guide"] == PD_GUIDE for variant in prepared.variants.values())
        else ""
    )
    result["next"] = location_caveat(
        f"Call experiment_start with the same design and expected_digest=request_digest.{review}",
        [variant.fields for variant in prepared.variants.values()],
    )
    return result


@tool(
    "experiment_start",
    "[S] Start an experiment as one Lens Evaluation workflow and return immediately; Lens runs it server-side: it "
    "deploys each variant, runs the scenarios (and optional baselines), and deletes the deployments unless "
    "preserve_deployment is set. This tool saves one configuration artifact per variant, creates the workflow "
    "(experiment id = Lens workflow id) and never waits or advances anything; follow it with experiment_get. "
    "Pass expected_digest from experiment_plan to refuse a changed plan. Stop it with experiment_cancel, rerun a "
    "failed one with experiment_retry, remove it with resource_delete. A design with prefill/decode deployments "
    "returns status awaiting_review and review_required: the user must review the plans, then call "
    "experiment_confirm.\n"
    "Side effects: creates configuration artifacts and a Lens evaluation workflow that deploys and benchmarks; "
    "resources: accelerators while it runs; duration: returns in seconds, the experiment takes minutes to hours; "
    "reversible: experiment_cancel.",
    StartExperimentInput,
    read_only=False,
    open_world=True,
)
async def start_experiment(ctx: ToolContext, args: StartExperimentInput) -> dict[str, Any]:
    design = ExperimentDesign(**{name: getattr(args, name) for name in ExperimentDesign.model_fields})
    return await ops.start(ctx, design, args.expected_digest)


def _not_found(experiment_id: str) -> ToolError:
    return ToolError(
        "not_found",
        f"No experiment '{experiment_id}'.",
        field="experiment_id",
        hint="Call resource_list(kinds=['experiment']) for ids.",
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
    if view.get("status") in ("cancelling", "awaiting_review"):
        return view["status"]
    return (view.get("active_case") or {}).get("status") or "queued"


@tool(
    "experiment_get",
    "[R] Status and results of an experiment. Returns status, progress (cases by state), the active case with its "
    "phase and heartbeat, last_progress_at, a stale verdict with reason, a heuristic ETA and poll_after_seconds. "
    "wait_seconds (max 50) really waits: wait_for='change' returns when progress changes, 'terminal' only when it "
    "finishes; the reply says whether it timed out. include=['results'] gives measurements per plan, scenario and "
    "repetition; 'comparison' adds deltas versus the first plan; 'points' adds every sweep point; 'cases' lists "
    "every case. Never changes the experiment.\n"
    "Side effects: none; resources: none; duration: up to wait_seconds; reversible: n/a.",
    GetExperimentInput,
    read_only=True,
)
async def get_experiment(ctx: ToolContext, args: GetExperimentInput) -> dict[str, Any]:
    try:
        workflow, active, waited, outcome = await watch(ctx, args.experiment_id, args.wait_seconds, args.wait_for)
    except LensApiError as error:
        if error.status in (404, 422):
            raise _not_found(args.experiment_id) from error
        raise
    view = summarize(ctx, workflow, active)
    remaining = await remaining_seconds(ctx, workflow)
    eta_seconds = remaining[0] if remaining else None
    if remaining:
        view["eta_remaining_seconds"] = {"value": eta_seconds, "basis": "mcp-heuristic"}
        if remaining[1]:
            view["eta_remaining_seconds"]["excludes"] = (
                f"benchmark time of {remaining[1]} unfinished case(s): Lens cannot estimate a workload without "
                "matrix and concurrency_stages, so only deployment time is counted for them."
            )
    view["poll_after_seconds"] = poll_after(_poll_phase(view), eta_seconds)
    if workflow.get("status") == "awaiting_review":
        view["review_required"] = ops.review_required(workflow)
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


# Open question for reviewers: this tool costs one slot of the core profile and
# only wraps Lens's confirm route. It is kept as its own tool so that an MCP
# client can require human approval for exactly the call that releases the GPUs.
# If the tool count matters more, drop it and leave confirmation to the Lens Web
# UI; experiment_start would still report awaiting_review, but an agent could no
# longer finish a prefill/decode experiment on its own.
@tool(
    "experiment_confirm",
    "[S] Confirm the reviewed deployment plans of an experiment that experiment_start or experiment_get reports as "
    "awaiting_review (it deploys prefill/decode DisaggregatedSets, which Lens only runs from reviewed plans). Pass "
    "every plan_id with its plan_digest from review_required; Lens refuses a changed or failed plan. Ask the user "
    "to review the plans first, in this conversation or in the Lens Web UI, unless they told you to go ahead. Then "
    "follow it with experiment_get. Only the creator can confirm.\n"
    "Side effects: queues the experiment, which then deploys and benchmarks; resources: accelerators while it runs; "
    "duration: returns in seconds; reversible: experiment_cancel.",
    ConfirmExperimentInput,
    read_only=False,
    open_world=True,
)
async def confirm_experiment(ctx: ToolContext, args: ConfirmExperimentInput) -> dict[str, Any]:
    return await ops.confirm(ctx, args.experiment_id, args.plan_digests, args.acknowledge_unknown)


@tool(
    "experiment_cancel",
    "[D] Cancel an experiment through Lens: the running benchmark is stopped and Lens deletes the deployments it "
    "created; finished measurements are kept. case_ids cancels only those cases. Returns the status, the cancelled "
    "case ids and the cleanup state (cleanup_pending means call it again). Cancelling a finished experiment is "
    "harmless. Only the creator can cancel.\n"
    "Side effects: aborts running benchmarks and deletes the experiment's deployments; resources: frees their "
    "accelerators; duration: up to ~2 min while Lens cleans up; reversible: no (experiment_retry reruns).",
    CancelExperimentInput,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def cancel_experiment(ctx: ToolContext, args: CancelExperimentInput) -> dict[str, Any]:
    return await ops.cancel(ctx, args.experiment_id, args.case_ids)


@tool(
    "experiment_retry",
    "[S] Retry a failed or cancelled experiment: Lens requeues its unfinished cases and runs them again with the "
    "same configuration (finished cases keep their results). Follow it with experiment_get. Only the creator can "
    "retry; other states return conflict_state.\n"
    "Side effects: requeues cases and redeploys; resources: accelerators while it runs; duration: returns in "
    "seconds; reversible: experiment_cancel.",
    RetryExperimentInput,
    read_only=False,
    open_world=True,
)
async def retry_experiment(ctx: ToolContext, args: RetryExperimentInput) -> dict[str, Any]:
    return await ops.retry(ctx, args.experiment_id)


@tool(
    "result_compare",
    "[R] Compare measurements across Lens experiments and standalone benchmark runs: median, min, max and the delta "
    "versus a baseline plan per scenario and concurrency or rate point, with SLA verdicts. Repetitions are "
    "grouped by default.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    CompareResultsInput,
    read_only=True,
)
async def compare_results(ctx: ToolContext, args: CompareResultsInput) -> dict[str, Any]:
    if not args.experiment_ids and not args.benchmark_run_ids:
        raise ToolError(
            "invalid_argument",
            "Pass experiment_ids or benchmark_run_ids.",
            field="experiment_ids",
            hint="Use ids from experiment_start or resource_list(kinds=['experiment']).",
        )
    rows: list[dict[str, Any]] = []
    for experiment_id in dict.fromkeys(args.experiment_ids):
        rows += case_rows(await _details(ctx, experiment_id), experiment_id)
    for run_id in dict.fromkeys(args.benchmark_run_ids):
        try:
            run = await ctx.api.request("GET", f"/evaluate/runs/{run_id}", timeout=60)
        except LensApiError as error:
            if error.status in (404, 422):
                raise ToolError(
                    "not_found",
                    f"No benchmark run '{run_id}'.",
                    field="benchmark_run_ids",
                    hint="Pass experiment ids as experiment_ids; benchmark_run_ids are standalone run ids.",
                ) from error
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


def _percentile(ordered: list[float], fraction: float) -> float:
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _summarise_artifact(content: Any) -> dict[str, Any]:
    """Bounded view of a report: per-field statistics for a list of records, the head of anything else."""
    records = content if isinstance(content, list) else None
    if records and all(isinstance(item, dict) for item in records):
        fields: dict[str, dict[str, float]] = {}
        for key in dict.fromkeys(key for item in records for key in item):
            values = sorted(
                item[key]
                for item in records
                if isinstance(item.get(key), int | float) and not isinstance(item.get(key), bool)
            )
            if values:
                fields[key] = {
                    "count": len(values),
                    "min": values[0],
                    "p50": _percentile(values, 0.5),
                    "p90": _percentile(values, 0.9),
                    "p99": _percentile(values, 0.99),
                    "max": values[-1],
                }
        return {"records": len(records), "numeric_fields": fields, "sample": records[:3]}
    text = json.dumps(content, default=str)
    return {"truncated": len(text) > MAX_ARTIFACT_CHARS, "content": text[:MAX_ARTIFACT_CHARS]}


@tool(
    "experiment_artifact_get",
    "[R] List the native benchmark reports saved for one case of an experiment, or read one: summary statistics "
    "for per-request records (count, min/p50/p90/p99/max per numeric field, three sample records) and the Lens "
    "download path for the full file. Never changes the experiment.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    GetArtifactInput,
    read_only=True,
)
async def get_experiment_artifact(ctx: ToolContext, args: GetArtifactInput) -> dict[str, Any]:
    workflow = await ops.load_workflow(ctx, args.experiment_id)
    case = next((item for item in workflow.get("cases") or [] if item.get("id") == args.case_id), None)
    if case is None:
        raise ToolError(
            "not_found",
            f"Experiment {args.experiment_id} has no case '{args.case_id}'.",
            field="case_id",
            hint="Call experiment_get(include=['cases']).",
        )
    run_id = case.get("evaluation_run_id")
    if not run_id:
        raise ToolError(
            "conflict_state",
            f"Case {args.case_id} has not started a benchmark run yet ({case.get('status')}).",
            field="case_id",
            hint="Poll experiment_get until the case has run.",
        )
    base = f"/evaluate/runs/{check_id(run_id, 'case_id')}/artifacts"
    result: dict[str, Any] = {"experiment_id": args.experiment_id, "case_id": args.case_id, "run_id": run_id}
    try:
        if args.name is None:
            return {**result, **await ctx.api.request("GET", base)}
        if ".." in args.name.split("/"):
            raise ToolError("invalid_argument", "name must not contain '..'.", field="name")
        content = await ctx.api.request("GET", f"{base}/{args.name}", timeout=120)
    except LensApiError as error:
        if error.status in (404, 422):
            raise ToolError(
                "not_found",
                f"No artifact '{args.name}' for case {args.case_id}.",
                field="name",
                hint="Call experiment_artifact_get without name to list the saved reports.",
            ) from error
        raise
    download = f"/api/v1/evaluate/runs/{run_id}/artifacts/{args.name}"
    return {**result, "name": args.name, "download_path": download, **_summarise_artifact(content)}


register_kind("experiment", list=ops.list_experiments, get=ops.get_entry, delete=ops.delete)
