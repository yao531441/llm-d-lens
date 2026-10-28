"""Experiments: run one workload against one or more deployment variants and compare the measurements."""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..errors import ToolError
from ..experiments_deploy import borrowed_target
from ..experiments_engine import (
    advance,
    build_view,
    delete_experiment,
    dependents_of_experiment,
    describe_experiment,
    initial_state,
    request_cancel,
    result_entries,
    wait_for_change,
)
from ..experiments_model import describe_workload
from ..experiments_plan import ExperimentDesign, check_specs, estimate, normalize
from ..experiments_results import DEFAULT_COMPARE_METRICS, comparability, compare_entries
from ..kinds import register_kind
from ..ledger import PLAN_TTL_SECONDS, Resource
from ..registry import tool

EXPERIMENT_ID = r"^exp-[a-z0-9]+$"
RESULT_REF = re.compile(r"^exp-[a-z0-9]+#[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
Include = Literal["results", "comparison", "points", "timeline"]
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


class PlanExperimentInput(ExperimentDesign):
    """The design; variants is required."""


class StartExperimentInput(ExperimentDesign):
    plan_id: str | None = Field(
        default=None,
        description="plan_id from plan_experiment: start exactly that design (do not also pass variants). "
        "Repeating the same plan_id returns the experiment it already started.",
    )
    client_request_id: str | None = Field(default=None, description="Retry key: repeating it returns the first result.")


class GetExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=EXPERIMENT_ID, description="exp- id from start_experiment.")
    include: list[Include] = Field(
        default_factory=list,
        description="results=compact table per variant and workload; comparison=deltas versus the first variant; "
        "points=every concurrency/rate point of sweeps; timeline=recent events.",
    )
    wait_seconds: int = Field(
        default=0, ge=0, le=25, description="Long poll: return early when the experiment changes or settles."
    )
    wait_until: Literal["change", "settled"] = Field(
        default="change", description="change=return on any progress; settled=only return when finished."
    )


class CancelExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=EXPERIMENT_ID, description="exp- id to cancel.")
    variant_ids: list[str] | None = Field(
        default=None, min_length=1, description="Cancel only these variants; default is the whole experiment."
    )
    keep_deployments: bool = Field(
        default=False,
        description="Whole-experiment cancel only: keep the temporary deployments (they become ordinary deployments "
        "you must delete). Default: delete them.",
    )
    reason: str | None = Field(default=None, max_length=200, description="Why it was cancelled (informational).")


class CompareResultsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_refs: list[str] = Field(
        min_length=2,
        max_length=20,
        description="Result references (exp-xxxx#variant/workload) from get_experiment(include=['results']).",
    )
    baseline_ref: str | None = Field(
        default=None, description="Reference that deltas are computed against; default the first."
    )
    metrics: list[Metric] | None = Field(
        default=None, description="Default: ttft p99, tpot p99, throughput and success rate."
    )
    normalize: Literal["none", "per_gpu"] = Field(
        default="none", description="per_gpu divides throughput by the accelerators each variant used."
    )


def _find_row(ctx: ToolContext, experiment_id: str) -> Resource:
    row = ctx.ledger.get(experiment_id)
    if row is None or row.kind != "experiment" or row.state == "deleted":
        raise ToolError(
            "not_found",
            f"No experiment '{experiment_id}'.",
            field="experiment_id",
            hint="Call list_resources(kinds=['experiment']) for ids.",
        )
    return row


def _design_view(design: dict[str, Any]) -> dict[str, Any]:
    return {
        "variants": [
            {
                key: variant.get(key)
                for key in ("variant_id", "label", "kind", "deployment_id", "spec", "gpus")
                if variant.get(key) not in (None, 0)
            }
            for variant in design["variants"]
        ],
        "workloads": [
            {"workload_id": w["workload_id"], "preset": w["preset"], "sla": w["sla"], **describe_workload(w)}
            for w in design["workloads"]
        ],
        "rigor": design["rigor"],
        "repetitions": design["repetitions"],
        "schedule": design["schedule"],
        "order": design["order"],
        "fairness": design["fairness"],
        "cache_condition": design["cache_condition"],
        "keep_deployments": design["keep_deployments"],
    }


@tool(
    "plan_experiment",
    "[R] Validate an experiment design without running anything: variants (a deployment spec the experiment creates "
    "temporarily, or an existing deployment you own that is only borrowed) x workloads x repetitions. Returns "
    "accelerator-hours, duration, fairness warnings and a plan_id for start_experiment. Routing-policy or "
    "other deployment-side ablations are expressed as separate spec variants. Use plan_deployment to check a "
    "single deployment instead.\n"
    "Side effects: none; resources: none; duration: synchronous, up to ~25 s (dry-runs each spec variant); "
    "reversible: n/a.",
    PlanExperimentInput,
    read_only=True,
    open_world=True,
)
async def plan_experiment(ctx: ToolContext, args: PlanExperimentInput) -> dict[str, Any]:
    design = await normalize(ctx, args)
    estimates, checks = await asyncio.gather(estimate(ctx, design), check_specs(ctx, design))
    errors, warnings = list(estimates["errors"]), list(estimates["warnings"])
    for check in checks:
        if check["status"] == "invalid":
            errors.append({"field": f"variants/{check['variant_id']}/spec", **check["error"]})
        elif check["status"] == "unchecked":
            warnings.append(
                {
                    "code": "UNCHECKED_PLAN",
                    "message": f"Variant '{check['variant_id']}' was not dry-run in time; it is planned at start.",
                }
            )
        for warning in check.get("warnings") or []:
            warnings.append({"code": "VARIANT_WARNING", "variant_id": check["variant_id"], **warning})
    total_runs = len(design["variants"]) * len(design["workloads"]) * design["repetitions"]
    result: dict[str, Any] = {
        "status": "invalid" if errors else "valid_with_warnings" if warnings else "valid",
        "design": _design_view(design),
        "benchmark_runs": total_runs,
        "resources": estimates["resources"],
        "timing": estimates["timing"],
        "validation": {"errors": errors, "warnings": warnings},
        "variant_checks": checks,
    }
    if errors:
        result["next"] = "Fix the errors and call plan_experiment again."
        return result
    result["plan_id"] = ctx.ledger.put_plan(
        owner=ctx.owner,
        cluster_id=design["cluster_id"],
        payload={"design": design, "unit_seconds": estimates["unit_seconds"]},
    )
    result["expires_in_seconds"] = PLAN_TTL_SECONDS
    result["next"] = "Call start_experiment(plan_id=...) to run it."
    return result


def _already_started(ctx: ToolContext, plan_id: str) -> Resource | None:
    rows = ctx.ledger.list_resources(owner=ctx.owner, kind="experiment", include_deleted=True)
    return next((row for row in rows if row.meta.get("plan_id") == plan_id), None)


@tool(
    "start_experiment",
    "[S] Start an experiment and return its exp- id immediately. It creates one temporary deployment per spec "
    "variant (each in its own Lens deployment run), runs the workloads against every variant one benchmark at a "
    "time, then deletes the temporary deployments; existing-deployment variants are borrowed and never modified "
    "or deleted. Pass the plan_id from plan_experiment or the same design inline. The benchmark runs server-side; "
    "progress advances whenever you poll get_experiment. Use cancel_experiment to stop it and "
    "delete_resources to remove it.\n"
    "Side effects: creates temporary deployments and Lens benchmark runs; resources: accelerators for each spec "
    "variant while it runs; duration: returns in seconds, the experiment takes minutes to hours; reversible: "
    "cancel_experiment.",
    StartExperimentInput,
    read_only=False,
    open_world=True,
)
async def start_experiment(ctx: ToolContext, args: StartExperimentInput) -> dict[str, Any]:
    plan_id = args.plan_id
    if plan_id:
        if args.variants is not None:
            raise ToolError(
                "invalid_argument",
                "Pass either plan_id or a design, not both.",
                field="plan_id",
                hint="Call start_experiment(plan_id=...) alone.",
            )
        existing = _already_started(ctx, plan_id)
        if existing is not None:
            return {"already_started": True, **build_view(existing, [])}
        plan = ctx.ledger.get_plan(plan_id, ctx.owner)
        if plan is None:
            raise ToolError(
                "plan_not_found",
                f"Plan '{plan_id}' does not exist or has expired.",
                field="plan_id",
                hint="Call plan_experiment again for a fresh plan_id.",
            )
        design, unit_seconds = plan["design"], plan["unit_seconds"]
        for variant in design["variants"]:
            if variant["kind"] == "existing":
                await borrowed_target(ctx, variant["deployment_id"])
    else:
        design = await normalize(ctx, args)
        estimates = await estimate(ctx, design)
        if estimates["errors"]:
            raise ToolError(
                "plan_invalid",
                "; ".join(error["message"] for error in estimates["errors"]),
                hint="Adjust the design, or call plan_experiment to inspect it.",
            )
        unit_seconds = estimates["unit_seconds"]
    row = ctx.ledger.add_resource(
        kind="experiment",
        owner=ctx.owner,
        cluster_id=design["cluster_id"],
        name=design["name"] or f"experiment-{uuid.uuid4().hex[:4]}",
        state="creating",
    )
    ctx.ledger.update_meta(row.id, initial_state(row.id, design, plan_id, unit_seconds))
    ctx.ledger.set_state(row.id, "active")
    # From here the experiment row exists, so a failed first step must not hide the exp- id: the next poll resumes it.
    first_step_error = None
    try:
        row = (await advance(ctx, row.id))[0] or row
    except Exception as error:  # noqa: BLE001
        first_step_error = str(error)[:300]
        row = ctx.ledger.get(row.id) or row
    view = {"started": True, **build_view(row, [])}
    if first_step_error:
        view["advance_error"] = first_step_error
        view["next"] = "Call get_experiment to resume the experiment; its state is persisted."
    return view


@tool(
    "get_experiment",
    "[R] Progress and results of an experiment. The state has four independent dimensions: execution "
    "(queued/running/succeeded/partially_succeeded/failed/cancelling/cancelled), measurement (valid runs), "
    "mechanism (server-side evidence) and cleanup (temporary deployments; residuals are listed if deletion "
    "failed). include=['results'] gives a table per variant and workload with TTFT, TPOT, ITL, latency, "
    "throughput and success rate; sweeps are summarised by their highest-throughput point (include 'points' for "
    "all). Polling also advances the experiment, so keep calling it with wait_seconds=25 until settled. Use "
    "compare_results to compare across experiments.\n"
    "Side effects: advances the experiment's own state; resources: none; duration: up to wait_seconds; "
    "reversible: n/a.",
    GetExperimentInput,
    read_only=True,
)
async def get_experiment(ctx: ToolContext, args: GetExperimentInput) -> dict[str, Any]:
    row = _find_row(ctx, args.experiment_id)
    if "phase" not in row.meta:
        raise ToolError("conflict_state", f"{row.id} is still being created.", hint="Retry in a few seconds.")
    mine = row.owner == ctx.owner
    contended = False
    if mine:
        advanced, contended = await wait_for_change(ctx, row.id, args.wait_seconds, args.wait_until)
        row = advanced or row
    return build_view(row, list(args.include), is_mine=mine, contended=contended)


@tool(
    "cancel_experiment",
    "[D] Cancel an experiment: stops the running benchmark, waits until Lens reports it terminal, then deletes the "
    "temporary deployments the experiment created (never borrowed ones). Results already measured are kept. "
    "Cancelling again is harmless. To remove the experiment record and its Lens benchmark runs as well use "
    "delete_resources.\n"
    "Side effects: aborts the benchmark run and deletes temporary deployments; resources: frees their "
    "accelerators; duration: up to ~20 s, longer teardown continues while you poll get_experiment; "
    "reversible: no.",
    CancelExperimentInput,
    read_only=False,
    destructive=True,
    idempotent=True,
)
async def cancel_experiment(ctx: ToolContext, args: CancelExperimentInput) -> dict[str, Any]:
    row = _find_row(ctx, args.experiment_id)
    if row.owner != ctx.owner:
        raise ToolError(
            "not_owner",
            f"{row.id} belongs to another user.",
            field="experiment_id",
            hint="You can read it with get_experiment but not cancel it.",
        )
    unknown = sorted(set(args.variant_ids or []) - set(row.meta["variants"]))
    if unknown:
        raise ToolError(
            "invalid_argument",
            f"Unknown variant ids: {', '.join(unknown)}.",
            field="variant_ids",
            hint="Use the variant_id values shown by get_experiment.",
        )
    row = (
        await request_cancel(
            ctx, row.id, variant_ids=args.variant_ids, keep_deployments=args.keep_deployments, reason=args.reason
        )
        or row
    )
    view = build_view(row, ["results"])
    view["cancelled"] = row.meta["phase"] == "settled"
    if not view["cancelled"]:
        view["next"] = "Teardown is still running; call get_experiment to follow it (it finishes while you poll)."
    return view


def _entry_for(ctx: ToolContext, ref: str) -> dict[str, Any]:
    if not RESULT_REF.match(ref):
        raise ToolError(
            "invalid_argument",
            f"'{ref}' is not a result reference.",
            field="result_refs",
            hint="Format: exp-xxxxxxxx#variant_id/workload_id.",
        )
    row = ctx.ledger.get(ref.split("#", 1)[0])
    if row is None or row.kind != "experiment" or row.state == "deleted" or "design" not in row.meta:
        raise ToolError(
            "not_found",
            f"No experiment for result reference '{ref}'.",
            field="result_refs",
            hint="Use references from get_experiment(include=['results']).",
        )
    entry = next((item for item in result_entries(row) if item["result_ref"] == ref), None)
    if entry is None:
        raise ToolError(
            "conflict_state",
            f"'{ref}' has no finished measurement (yet, or its runs failed).",
            field="result_refs",
            hint="Check get_experiment(include=['results']) for the references that exist.",
        )
    return entry


@tool(
    "compare_results",
    "[R] Compare measured results across experiments or within one: deltas of TTFT, TPOT, throughput and success "
    "rate versus a baseline, with comparability checks (same cluster, model, accelerator type, workload, "
    "accelerator count, benchmark harness version). Mismatches are reported as issues, not hidden. For one "
    "experiment's own comparison use get_experiment(include=['comparison']).\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    CompareResultsInput,
    read_only=True,
)
async def compare_results(ctx: ToolContext, args: CompareResultsInput) -> dict[str, Any]:
    refs = list(dict.fromkeys(args.result_refs))
    if len(refs) < 2:
        raise ToolError("invalid_argument", "Pass at least two different result references.", field="result_refs")
    baseline = args.baseline_ref or refs[0]
    if baseline not in refs:
        raise ToolError(
            "invalid_argument",
            "baseline_ref must be one of result_refs.",
            field="baseline_ref",
            hint="Add the baseline to result_refs or pick one of them.",
        )
    entries = [_entry_for(ctx, ref) for ref in refs]
    per_gpu = args.normalize == "per_gpu"
    return {
        "comparability": comparability(entries, per_gpu),
        "comparison": compare_entries(entries, baseline, list(args.metrics or DEFAULT_COMPARE_METRICS), per_gpu),
        "results": [
            {key: entry[key] for key in ("result_ref", "label", "metrics", "repetitions", "gpus", "sla")}
            for entry in entries
        ],
    }


register_kind("experiment", delete=delete_experiment, describe=describe_experiment, dependents=dependents_of_experiment)
