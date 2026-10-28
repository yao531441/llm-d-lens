"""Validation and estimation of an experiment design, shared by plan_experiment and start_experiment."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .cluster_info import cluster_summary
from .context import ToolContext
from .errors import ToolError
from .experiments_deploy import borrowed_target, plan_spec, row_gpus, spec_gpus
from .experiments_model import (
    ExperimentVariant,
    SlaTargets,
    VariantSpec,
    WorkloadSpec,
    resolve_workloads,
)
from .lens_api import LensApiError

REPETITIONS = {"quick": 1, "standard": 3, "report": 5}
DEPLOY_SECONDS_ASSUMED = 600
PLANNING_BUDGET_SECONDS = 22
_TIMING_FIELDS = ("matrix", "concurrency_stages", "shared_prefix", "warmup_requests")


class ExperimentDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120, description="Experiment name; generated.")
    cluster_id: str | None = Field(
        default=None,
        description="Cluster for spec variants (from list_clusters). Optional when every variant is an existing "
        "deployment; all variants must be on the same cluster.",
    )
    common_spec: VariantSpec | None = Field(
        default=None,
        description="Defaults merged under every spec variant so variants differ only in what is being compared.",
    )
    variants: list[ExperimentVariant] | None = Field(
        default=None, min_length=1, max_length=12, description="Configurations to measure (required without plan_id)."
    )
    workloads: list[WorkloadSpec] | None = Field(
        default=None,
        min_length=1,
        max_length=10,
        description="Load profiles; every variant runs every workload. Default: one 'chat' preset.",
    )
    sla: SlaTargets | None = Field(default=None, description="Pass/fail targets for every workload.")
    rigor: Literal["quick", "standard", "report"] = Field(
        default="quick",
        description="Repetitions of each (variant, workload): quick=1, standard=3, report=5. Results are medians.",
    )
    schedule: Literal["sequential", "parallel"] = Field(
        default="sequential",
        description="sequential=deploy one spec variant at a time (fewest accelerators, no interference); "
        "parallel=deploy all at once (faster, peak accelerators = sum). Benchmarks always run one at a time.",
    )
    order: Literal["sequential", "randomized", "interleaved"] = Field(
        default="sequential",
        description="Run order: sequential=variants as listed; randomized=shuffled variant order (reduces order "
        "bias); interleaved=alternate variants for every repetition (needs schedule='parallel').",
    )
    fairness: Literal["per_configuration", "equal_gpu_budget"] = Field(
        default="per_configuration",
        description="per_configuration=each variant is measured as configured, unequal accelerator counts only "
        "warn; equal_gpu_budget=refuse designs where variants use different accelerator counts.",
    )
    warmup_requests: int = Field(default=2, ge=0, le=50, description="Uncounted warm-up requests before sweeps.")
    cache_condition: Literal["as_is", "cold"] = Field(
        default="as_is",
        description="cold=every spec variant starts on fresh pods (always true for spec variants); Lens cannot "
        "flush the cache of a borrowed deployment, so cold with a deployment_id variant only warns.",
    )
    keep_deployments: bool = Field(
        default=False,
        description="Keep the temporary deployments after the experiment (they become ordinary deployments "
        "you own and must delete). Default: delete them when each variant finishes.",
    )
    ttl_hours: int = Field(
        default=8, ge=1, le=72, description="Lifetime recorded for the experiment and its deployments."
    )
    max_gpu_hours: float | None = Field(
        default=None, gt=0, description="Refuse the design when the accelerator-hour estimate exceeds this."
    )


def _merge(common: VariantSpec | None, spec: VariantSpec) -> dict[str, Any]:
    return {**(common.model_dump(exclude_none=True) if common else {}), **spec.model_dump(exclude_none=True)}


async def normalize(ctx: ToolContext, design: ExperimentDesign) -> dict[str, Any]:
    """Structural and ownership validation; returns the JSON-serialisable design the engine runs."""
    if not design.variants:
        raise ToolError(
            "invalid_argument", "variants is required.", field="variants", hint="Pass 1-12 variants or a plan_id."
        )
    seen: set[str] = set()
    clusters: set[str] = set()
    variants = []
    for index, variant in enumerate(design.variants):
        if variant.variant_id in seen:
            raise ToolError(
                "invalid_argument",
                f"Duplicate variant_id '{variant.variant_id}'.",
                field=f"variants/{index}/variant_id",
                hint="Variant ids must be unique within the experiment.",
            )
        seen.add(variant.variant_id)
        if variant.spec is not None:
            spec = _merge(design.common_spec, variant.spec)
            missing = [name for name in ("guide", "model") if name not in spec]
            if missing:
                raise ToolError(
                    "invalid_argument",
                    f"Variant '{variant.variant_id}' has no {' or '.join(missing)} (set it in spec or common_spec).",
                    field=f"variants/{index}/spec",
                    hint="Every spec variant needs a guide and a model.",
                )
            if not design.cluster_id:
                raise ToolError(
                    "invalid_argument",
                    "cluster_id is required for spec variants.",
                    field="cluster_id",
                    hint="Call list_clusters and pass a cluster id.",
                )
            variants.append(
                {
                    "variant_id": variant.variant_id,
                    "label": variant.label,
                    "kind": "spec",
                    "spec": spec,
                    "gpus": spec_gpus(spec),
                }
            )
        else:
            row, _ = await borrowed_target(ctx, variant.deployment_id)
            clusters.add(row.cluster_id)
            variants.append(
                {
                    "variant_id": variant.variant_id,
                    "label": variant.label or row.name,
                    "kind": "existing",
                    "deployment_id": row.id,
                    "gpus": row_gpus(row),
                }
            )
    if design.cluster_id:
        clusters.add(design.cluster_id)
    if len(clusters) > 1:
        raise ToolError(
            "unsupported_combination",
            f"Variants span several clusters: {', '.join(sorted(clusters))}.",
            field="cluster_id",
            hint="Run one experiment per cluster and use compare_results to combine them.",
        )
    if design.order == "interleaved" and design.schedule != "parallel":
        raise ToolError(
            "invalid_argument",
            "order='interleaved' needs every variant deployed at once.",
            field="order",
            hint="Set schedule='parallel', or use order='randomized'.",
        )
    if (
        design.keep_deployments
        and design.schedule == "sequential"
        and sum(1 for v in variants if v["kind"] == "spec") > 1
    ):
        raise ToolError(
            "unsupported_combination",
            "keep_deployments with several spec variants needs them all deployed at once.",
            field="keep_deployments",
            hint="Set schedule='parallel', or keep_deployments=false.",
        )
    return {
        "name": design.name,
        "cluster_id": clusters.pop(),
        "variants": variants,
        "workloads": resolve_workloads(design.workloads or [_default_workload()], design.sla, design.warmup_requests),
        "rigor": design.rigor,
        "repetitions": REPETITIONS[design.rigor],
        "schedule": design.schedule,
        "order": design.order,
        "order_seed": random.randrange(2**31),  # noqa: S311  (ordering, not security)
        "fairness": design.fairness,
        "cache_condition": design.cache_condition,
        "keep_deployments": design.keep_deployments,
        "ttl_hours": design.ttl_hours,
        "max_gpu_hours": design.max_gpu_hours,
    }


def _default_workload() -> WorkloadSpec:
    return WorkloadSpec(preset="chat")


async def _estimate_workload(ctx: ToolContext, workload: dict[str, Any]) -> tuple[int, int] | None:
    body = {key: workload["benchmark"][key] for key in _TIMING_FIELDS if key in workload["benchmark"]}
    try:
        timing = await ctx.api.request("POST", "/evaluate/timing-estimate", body=body)
    except LensApiError:
        return None
    if timing.get("lower_seconds") is None or timing.get("upper_seconds") is None:
        return None
    return timing["lower_seconds"], timing["upper_seconds"]


async def estimate(ctx: ToolContext, design: dict[str, Any]) -> dict[str, Any]:
    """Timing, accelerator-hours, capacity and fairness findings for a normalized design (read-only)."""
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    timings = await asyncio.gather(*(_estimate_workload(ctx, w) for w in design["workloads"]))
    if any(timing is None for timing in timings):
        warnings.append(
            {
                "code": "TIMING_UNKNOWN",
                "message": "Lens could not estimate some workload durations; totals use only the known ones.",
            }
        )
    per_variant_bench = sum(t[1] for t in timings if t) * design["repetitions"]
    spec_variants = [v for v in design["variants"] if v["kind"] == "spec"]
    deploy = DEPLOY_SECONDS_ASSUMED
    benchmark_total = per_variant_bench * len(design["variants"])
    if design["schedule"] == "sequential":
        total_seconds = sum(per_variant_bench + (deploy if v["kind"] == "spec" else 0) for v in design["variants"])
        peak_gpus = max((v["gpus"] for v in spec_variants), default=0)
        gpu_seconds = sum(v["gpus"] * (deploy + per_variant_bench) for v in spec_variants)
    else:
        total_seconds = benchmark_total + (deploy if spec_variants else 0)
        peak_gpus = sum(v["gpus"] for v in spec_variants)
        gpu_seconds = peak_gpus * total_seconds
    gpu_hours = round(gpu_seconds / 3600, 2)
    if design["max_gpu_hours"] and gpu_hours > design["max_gpu_hours"]:
        errors.append(
            {
                "code": "MAX_GPU_HOURS_EXCEEDED",
                "message": f"Estimated {gpu_hours} accelerator-hours exceeds max_gpu_hours={design['max_gpu_hours']}.",
            }
        )
    budgets = {v["gpus"] for v in design["variants"] if v["gpus"]}
    if len(budgets) > 1:
        finding = {
            "code": "UNEQUAL_GPU_BUDGET",
            "message": f"Variants use different accelerator counts ({sorted(budgets)}); compare per_gpu or equalise.",
        }
        (errors if design["fairness"] == "equal_gpu_budget" else warnings).append(finding)
    if any(v["kind"] == "existing" for v in design["variants"]):
        if len(design["variants"]) > 1 and spec_variants:
            warnings.append(
                {
                    "code": "BORROWED_STATE",
                    "message": "Existing deployments keep their warm caches and prior load; spec variants start cold.",
                }
            )
        if design["cache_condition"] == "cold":
            warnings.append(
                {"code": "COLD_NOT_GUARANTEED", "message": "Lens cannot flush the cache of an existing deployment."}
            )
    if design["repetitions"] == 1 and len(design["variants"]) > 1:
        warnings.append(
            {
                "code": "SINGLE_RUN",
                "message": "rigor='quick' measures each configuration once, so noise is unknown; use 'standard'.",
            }
        )
    free, fits = await _fits_now(ctx, design, peak_gpus)
    if fits is False:
        warnings.append(
            {"code": "INSUFFICIENT_CAPACITY_NOW", "message": f"Needs {peak_gpus} accelerators, {free} are free now."}
        )
    return {
        "resources": {
            "schedule": design["schedule"],
            "peak_gpus": peak_gpus,
            "gpu_hours_estimate": gpu_hours,
            "fits_now": fits,
        },
        "timing": {
            "estimated_seconds": total_seconds,
            "per_workload_seconds": {
                w["workload_id"]: list(t) if t else None for w, t in zip(design["workloads"], timings, strict=True)
            },
            "assumptions": f"deployment takes {deploy}s per spec variant; benchmark uses Lens's upper estimate",
        },
        "errors": errors,
        "warnings": warnings,
        "unit_seconds": {
            w["workload_id"]: t[1] if t else None for w, t in zip(design["workloads"], timings, strict=True)
        },
    }


async def _fits_now(ctx: ToolContext, design: dict[str, Any], peak_gpus: int) -> tuple[int | None, bool | None]:
    if not peak_gpus:
        return None, True
    try:
        free = (await cluster_summary(ctx, design["cluster_id"]))["accelerators_free"]
    except LensApiError:
        return None, None
    return free, None if free is None else free >= peak_gpus


async def check_specs(ctx: ToolContext, design: dict[str, Any]) -> list[dict[str, Any]]:
    """Dry-run the deployment planner for each spec variant, within a time budget (planning is slow)."""
    spec_variants = [v for v in design["variants"] if v["kind"] == "spec"]
    gate = asyncio.Semaphore(3)

    async def one(variant: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            try:
                plan = await plan_spec(ctx, design["cluster_id"], variant["spec"])
            except ToolError as error:
                return {"variant_id": variant["variant_id"], "status": "invalid", "error": error.payload()["error"]}
            except LensApiError as error:
                return {
                    "variant_id": variant["variant_id"],
                    "status": "invalid",
                    "error": {"code": error.code() or f"lens_{error.status}", "message": error.message()},
                }
            ctx.ledger.drop_plan(plan["plan_id"])
            return {
                "variant_id": variant["variant_id"],
                "status": "valid",
                "warnings": (plan.get("validation") or {}).get("warnings") or [],
                "accelerators_free": (plan.get("resources") or {}).get("accelerators_free"),
            }

    tasks = {asyncio.ensure_future(one(v)): v for v in spec_variants}
    done, pending = await asyncio.wait(tasks, timeout=PLANNING_BUDGET_SECONDS)
    for task in pending:
        task.cancel()
    results = [task.result() for task in done]
    results += [{"variant_id": tasks[task]["variant_id"], "status": "unchecked"} for task in pending]
    return sorted(results, key=lambda item: item["variant_id"])
