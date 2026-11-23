"""Dry run of an experiment: per variant guide planning and render, per scenario timing, budget and digest."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

from llm_d_lens.deploy.contracts import DeployableConfiguration
from llm_d_lens.evaluation.router import baseline_configuration

from .cluster_info import cluster_summary
from .context import ToolContext
from .errors import ToolError
from .evaluation_design import (
    LARGE_EXPERIMENT_PLANS,
    MAX_BASELINES,
    MAX_PLANS,
    MAX_SCENARIOS,
    ExperimentDesign,
    PlanSpec,
    canonical,
    check_limits,
    expand,
    predicted_cases,
    resolve_seed,
)
from .guide_fields import PD_GUIDE, merge_fields, normalize_fields
from .planning import VariantPlan, plan_configuration, slug

DEPLOY_SECONDS, PD_EXTRA_DEPLOY_SECONDS = 120, 60
MAX_LISTED_CASES = 100


@dataclass
class Prepared:
    design: ExperimentDesign
    seed: int
    cluster_id: str
    session_id: str
    plans: list[PlanSpec]
    variants: dict[str, VariantPlan]
    cases: list[dict[str, Any]]
    timing: dict[str, dict[str, Any]]
    free_gpus: int | None
    digest: str
    baseline_gpus: dict[tuple[str, str], int] = field(default_factory=dict)  # (variant id, baseline type) -> GPUs


async def resolve_cluster(ctx: ToolContext, cluster_id: str | None) -> str:
    if cluster_id:
        return cluster_id
    data = await ctx.api.request("GET", "/cluster/clusters")
    ids = [item["id"] for item in data.get("items") or []]
    if len(ids) == 1:
        return ids[0]
    raise ToolError(
        "invalid_argument",
        "cluster_id is required." + (f" Clusters: {', '.join(ids)}." if ids else ""),
        field="cluster_id",
        hint="Call cluster_list.",
    )


async def _plan_variant(
    ctx: ToolContext,
    variant_id: str,
    fields: dict[str, Any],
    *,
    cluster_id: str,
    cluster: dict[str, Any],
    session_id: str,
    summary: dict[str, Any],
    catalog: dict[str, Any],
    deployment_name: str | None,
    model_cache_volume_id: str | None = None,
) -> VariantPlan:
    name = deployment_name or f"mcp-{slug(fields['model'].rsplit('/', 1)[-1], 30)}-{slug(variant_id, 30)}"
    return await plan_configuration(
        ctx,
        variant_id,
        fields,
        cluster_id=cluster_id,
        cluster=cluster,
        session_id=session_id,
        summary=summary,
        catalog=catalog,
        deployment_name=name,
        model_cache_volume_id=model_cache_volume_id,
    )


async def _timing(ctx: ToolContext, scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    unique = {canonical(s.get("benchmark") or {}): s.get("benchmark") or {} for s in scenarios}
    keys = list(unique)
    results = await asyncio.gather(
        *(ctx.api.request("POST", "/evaluate/timing-estimate", body=unique[key]) for key in keys)
    )
    return dict(zip(keys, results, strict=True))


def _baseline_gpus(design: ExperimentDesign, variants: dict[str, VariantPlan]) -> dict[tuple[str, str], int]:
    """GPUs of each independent baseline, sized by the same rule Lens applies when it builds the baseline."""
    sizes: dict[tuple[str, str], int] = {}
    for variant_id, item in variants.items():
        for baseline_type in design_baselines(design):
            parameters = design.baseline_parameters.get(baseline_type)
            try:
                content = baseline_configuration(
                    DeployableConfiguration.model_validate(item.configuration),
                    baseline_type,
                    [],
                    "",
                    parameters.model_dump(exclude_none=True) if parameters else None,
                ).content
            except ValueError as error:
                raise ToolError(
                    "plan_invalid",
                    f"Variant '{variant_id}', baseline '{baseline_type}': {error}",
                    field=f"baseline_parameters/{baseline_type}",
                    hint="Set baseline_parameters.replicas or tensor_parallel_size explicitly.",
                ) from error
            decode = content.get("decode") or {}
            sizes[(variant_id, baseline_type)] = (
                int(decode.get("replicaCount") or 1) * int(decode.get("tensorParallelSize") or 1)
                if decode
                else item.gpus
            )
    return sizes


def design_baselines(design: ExperimentDesign) -> list[str]:
    """Baselines that get a deployment of their own; kubernetes-service reuses the guide deployment."""
    return [name for name in design.baseline_types if name != "kubernetes-service"]


def _digest(design: ExperimentDesign, seed: int, variants: dict[str, VariantPlan], cluster_id: str) -> str:
    key_fields = {
        variant_id: {
            "accelerator": item.accelerator,
            "image": item.image,
            "gpus": item.gpus,
            "guide_variant": item.guide_variant,
            "checksum": item.configuration.get("checksum"),
        }
        for variant_id, item in sorted(variants.items())
    }
    payload = {
        "design": design.model_dump(by_alias=True, exclude={"seed"}, mode="json"),
        "seed": seed,
        "cluster_id": cluster_id,
        "variants": key_fields,
    }
    return "sha256:" + hashlib.sha256(canonical(payload).encode()).hexdigest()[:32]


async def prepare(ctx: ToolContext, design: ExperimentDesign) -> Prepared:
    check_limits(design)
    seed = resolve_seed(design)
    cluster_id = await resolve_cluster(ctx, design.cluster_id)
    names = {variant.id: variant.deployment_name for variant in design.variants}
    merged = {
        variant.id: normalize_fields(merge_fields(design.common, variant.plan), variant.id)
        for variant in design.variants
    }
    needs_catalog = any(
        (fields.get("source") or {}).get("mode", "official") == "official" for fields in merged.values()
    )
    summary, session, catalog = await asyncio.gather(
        cluster_summary(ctx, cluster_id),
        ctx.api.request("GET", "/cluster/session", params={"clusterId": cluster_id}),
        ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": cluster_id})
        if needs_catalog
        else asyncio.sleep(0, result={}),
    )
    session_id, cluster = session["sessionId"], session["cluster"]
    planned = await asyncio.gather(
        *(
            _plan_variant(
                ctx,
                variant_id,
                fields,
                cluster_id=cluster_id,
                cluster=cluster,
                session_id=session_id,
                summary=summary,
                catalog=catalog,
                deployment_name=names[variant_id],
                model_cache_volume_id=design.model_cache_volume_id,
            )
            for variant_id, fields in merged.items()
        )
    )
    variants = {item.variant_id: item for item in planned}
    plans = expand(design, seed)
    timing = await _timing(ctx, [scenario for plan in plans[:1] for scenario in plan.scenarios])
    return Prepared(
        design,
        seed,
        cluster_id,
        session_id,
        plans,
        variants,
        predicted_cases(design, plans),
        timing,
        summary.get("accelerators_free"),
        _digest(design, seed, variants, cluster_id),
        _baseline_gpus(design, variants),
    )


def deployment_gpus(prepared: Prepared, plan_id: str, baseline: str | None) -> int:
    variant_id = next(plan.variant_id for plan in prepared.plans if plan.plan_id == plan_id)
    return prepared.baseline_gpus[(variant_id, baseline)] if baseline else prepared.variants[variant_id].gpus


def peak_gpus(prepared: Prepared) -> int:
    """Deployments run one after another, or all stay up when ``preserve_deployment`` keeps them."""
    sizes = [deployment_gpus(prepared, plan_id, baseline) for plan_id, baseline in deployments(prepared)]
    return sum(sizes) if prepared.design.preserve_deployment else max(sizes)


def _case_seconds(
    prepared: Prepared, scenarios: dict[str, dict[str, Any]], case: dict[str, Any]
) -> tuple[int, int] | None:
    """Lens' benchmark estimate for one case; ``None`` when Lens cannot estimate this workload."""
    estimate = prepared.timing.get(canonical(scenarios[case["scenario"]].get("benchmark") or {})) or {}
    if estimate.get("basis") == "unknown":
        return None
    return int(estimate.get("lower_seconds") or 0), int(estimate.get("upper_seconds") or 0)


def eta(prepared: Prepared) -> dict[str, Any]:
    by_plan = {plan.plan_id: {s["id"]: s for s in plan.scenarios} for plan in prepared.plans}
    lower = upper = 0
    unknown_cases = 0
    for case in prepared.cases:
        seconds = _case_seconds(prepared, by_plan[case["plan"]], case)
        if seconds is None:
            unknown_cases += 1
            continue
        lower, upper = lower + seconds[0], upper + seconds[1]
    variant_of = {plan.plan_id: prepared.variants[plan.variant_id] for plan in prepared.plans}
    deploy = sum(
        DEPLOY_SECONDS
        + (PD_EXTRA_DEPLOY_SECONDS if baseline is None and variant_of[plan_id].fields["guide"] == PD_GUIDE else 0)
        for plan_id, baseline in deployments(prepared)
    )
    if unknown_cases:
        return {
            "benchmark": {"lower": None, "upper": None},
            "deploy": deploy,
            "basis": "deploy-heuristic-only",
            "unknown": f"Lens cannot estimate the benchmark time of {unknown_cases} of {len(prepared.cases)} case(s): "
            "the scenario has no matrix together with concurrency_stages (or a shared_prefix workload), so its "
            "length depends on harness defaults at run time.",
            "how_to_check": "Give each scenario a matrix and concurrency_stages to get an estimate, or follow the "
            "experiment with experiment_get once it runs.",
        }
    return {
        "benchmark": {"lower": lower, "upper": upper},
        "deploy": deploy,
        "basis": "lens-timing-estimate+mcp-heuristic",
    }


def deployments(prepared: Prepared) -> list[tuple[str, str | None]]:
    """One deployment per plan for the guide, one per non-same-pod baseline type; same-pod baselines reuse it."""
    found = {
        (case["plan"], None if case.get("baseline_type") in (None, "kubernetes-service") else case["baseline_type"])
        for case in prepared.cases
    }
    return sorted(found, key=lambda item: (item[0], item[1] or ""))


def warnings(prepared: Prepared) -> list[dict[str, Any]]:
    """Identical messages are returned once with the plans they affect."""
    collected: dict[tuple[str, str], list[str]] = {}

    def add(code: str, message: str, affects: list[str]) -> None:
        bucket = collected.setdefault((code, message), [])
        bucket.extend(item for item in affects if item not in bucket)

    design = prepared.design
    if design.repetitions.mode == "same_deployment" and design.repetitions.count > 1:
        add(
            "REPEATS_SHARE_DEPLOYMENT",
            "Repetitions run back to back on one deployment, so later runs may benefit from warm prefix caches.",
            [plan.plan_id for plan in prepared.plans],
        )
    if len(prepared.plans) > LARGE_EXPERIMENT_PLANS:
        add(
            "LARGE_EXPERIMENT",
            f"{len(prepared.plans)} plans deploy one after another; the experiment may take many hours.",
            [],
        )
    peak = peak_gpus(prepared)
    if prepared.free_gpus is not None and peak > prepared.free_gpus:
        add(
            "INSUFFICIENT_CAPACITY",
            f"The largest deployment needs {peak} accelerators but {prepared.free_gpus} are free now; "
            "Lens checks capacity when the experiment starts.",
            [],
        )
    for variant_id, item in prepared.variants.items():
        for text in item.warnings:
            add("GUIDE_WARNING", text, [variant_id])
    for estimate in prepared.timing.values():
        if estimate.get("timeout_warning"):
            add("TIMEOUT_TOO_SHORT", estimate["timeout_warning"], [])
    return [{"code": code, "message": message, "affects": affects} for (code, message), affects in collected.items()]


def cases_breakdown(prepared: Prepared) -> dict[str, Any]:
    """Why cases_total is not simply variants x scenarios x repetitions."""
    baselines: dict[str, int] = {}
    for case in prepared.cases:
        if case["kind"] == "baseline":
            baselines[case["baseline_type"]] = baselines.get(case["baseline_type"], 0) + 1
    breakdown: dict[str, Any] = {
        "guide": sum(1 for case in prepared.cases if case["kind"] == "guide"),
        "baseline": baselines,
    }
    if baselines:
        breakdown["note"] = (
            "Guide cases are plans x scenarios. A baseline other than kubernetes-service runs once per plan and "
            "distinct benchmark, not once per guide case; kubernetes-service runs after every guide case. "
            "Concurrency points inside a scenario are not separate cases."
        )
    return breakdown


def describe(prepared: Prepared) -> dict[str, Any]:
    peak = peak_gpus(prepared)
    times = eta(prepared)
    runs = deployments(prepared)
    gpu_hours = None
    if times["benchmark"]["upper"] is not None:
        per_deployment = (times["benchmark"]["upper"] + times["deploy"]) / max(1, len(runs))
        gpu_hours = round(
            sum(deployment_gpus(prepared, plan_id, baseline) * per_deployment for plan_id, baseline in runs) / 3600, 1
        )
    return {
        "cluster_id": prepared.cluster_id,
        "seed": prepared.seed,
        "order": prepared.design.order,
        "plans": [
            {
                "plan_id": plan.plan_id,
                "variant_id": plan.variant_id,
                "repetition": plan.repetition,
                "guide": prepared.variants[plan.variant_id].fields["guide"],
                "model": prepared.variants[plan.variant_id].fields["model"],
                "accelerator": prepared.variants[plan.variant_id].accelerator,
                "image": prepared.variants[plan.variant_id].image,
                "source": prepared.variants[plan.variant_id].source,
                "applied_custom_patches": prepared.variants[plan.variant_id].applied_patches,
                "gpus": prepared.variants[plan.variant_id].gpus,
                "scenarios": [scenario["id"] for scenario in plan.scenarios],
            }
            for plan in prepared.plans
        ],
        "case_order": prepared.cases[:MAX_LISTED_CASES],
        "case_order_truncated": len(prepared.cases) > MAX_LISTED_CASES,
        "cases_total": len(prepared.cases),
        "cases_breakdown": cases_breakdown(prepared),
        "deployments_total": len(runs),
        "peak_gpus": peak,
        "gpu_hours_estimate": gpu_hours,
        "eta_seconds": times,
        "limits": {"plans": MAX_PLANS, "scenarios_per_plan": MAX_SCENARIOS, "baseline_types": MAX_BASELINES},
        "fits_now": None if prepared.free_gpus is None else peak <= prepared.free_gpus,
        "warnings": warnings(prepared),
        "request_digest": prepared.digest,
    }
