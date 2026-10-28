"""Dry run of an experiment: per variant guide planning and render, per scenario timing, budget and digest."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

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
from .guide_fields import PD_GUIDE, candidate_config, gpu_count, merge_fields, normalize_fields, plan_request
from .planning import cached_volume, pick_accelerator, resolve_guide, serving_image, slug

DEPLOY_SECONDS, PD_EXTRA_DEPLOY_SECONDS = 120, 60
MAX_LISTED_CASES = 100


@dataclass
class VariantPlan:
    variant_id: str
    fields: dict[str, Any]
    accelerator: str
    model_server: str
    guide_variant: str
    image: str
    gpus: int
    configuration: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


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
        hint="Call list_clusters.",
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
) -> VariantPlan:
    accelerator = pick_accelerator(fields.get("accelerator"), summary)
    server, guide_variant = resolve_guide(
        catalog, fields["guide"], accelerator, fields.get("modelServer"), fields.get("guideVariant")
    )
    if not fields.get("modelSource") and not fields.get("modelPath"):
        volume = await cached_volume(ctx, cluster_id, fields["model"], None)
        fields = {
            **fields,
            "modelSource": "auto-cache",
            "modelPvcClaimName": volume.get("pvcName") or "",
            "modelStorageKind": volume.get("kind") or "",
            "storageVolumeId": volume["id"],
        }
    plan = await ctx.api.request(
        "POST",
        "/guide-planning/plan",
        body=plan_request(
            fields, accelerator=accelerator, model_server=server, guide_variant=guide_variant, session_id=session_id
        ),
        timeout=240,
    )
    validation = plan.get("validation") or {}
    if plan.get("status") not in ("VERIFIED", "PLANNED") or validation.get("errors"):
        raise ToolError(
            "plan_invalid",
            f"Variant '{variant_id}': "
            + ("; ".join(validation.get("errors") or []) or f"guide planning returned {plan.get('status')}."),
            field=f"variants/{variant_id}/plan",
            hint="Adjust the variant's fields; Lens validation messages are quoted above.",
        )
    manifest = (plan.get("plannedDeployment") or {}).get("content")
    if not manifest:
        raise ToolError("plan_empty", f"Variant '{variant_id}': the guide produced an empty manifest.")
    image = fields.get("runtimeImage") or serving_image(
        manifest, (plan.get("deployment") or {}).get("readinessDeployments") or []
    )
    if not image:
        raise ToolError(
            "image_unresolved",
            f"Variant '{variant_id}': could not determine the model-server image.",
            field=f"variants/{variant_id}/plan/runtimeImage",
            hint="Set runtimeImage explicitly.",
        )
    name = deployment_name or f"mcp-{slug(fields['model'].rsplit('/', 1)[-1], 30)}-{slug(variant_id, 30)}"
    rendered = await ctx.api.request(
        "POST",
        "/configurations/render",
        body={
            "candidate_config": candidate_config(
                fields, cluster=cluster, model_server=server, guide_variant=guide_variant, image=image
            ),
            "render": {
                "format": "manifest",
                "guide_ref": fields["guide"],
                "template_ref": plan["source"]["files"][0],
                "guide_source": plan["source"],
                "rendered_manifest": manifest,
                "deployment_bundle": plan.get("deploymentBundle"),
                "cluster_ref": {
                    "id": cluster["id"],
                    "name": cluster["name"],
                    "session_id": session_id,
                    "connection": "managed-kubeconfig",
                },
                "deployment": plan["deployment"],
                "deploymentName": name,
                "modelSecret": {"mode": "none"},
            },
        },
    )
    return VariantPlan(
        variant_id,
        fields,
        accelerator,
        server,
        guide_variant,
        image,
        gpu_count(fields),
        rendered["deployable_configuration"],
        list(validation.get("warnings") or []),
    )


async def _timing(ctx: ToolContext, scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    unique = {canonical(s.get("benchmark") or {}): s.get("benchmark") or {} for s in scenarios}
    keys = list(unique)
    results = await asyncio.gather(
        *(ctx.api.request("POST", "/evaluate/timing-estimate", body=unique[key]) for key in keys)
    )
    return dict(zip(keys, results, strict=True))


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
    summary, session, catalog = await asyncio.gather(
        cluster_summary(ctx, cluster_id),
        ctx.api.request("GET", "/cluster/session", params={"clusterId": cluster_id}),
        ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": cluster_id}),
    )
    session_id, cluster = session["sessionId"], session["cluster"]
    names = {variant.id: variant.deployment_name for variant in design.variants}
    merged = {
        variant.id: normalize_fields(merge_fields(design.common, variant.plan), variant.id)
        for variant in design.variants
    }
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
    )


def peak_gpus(prepared: Prepared) -> int:
    sizes = [prepared.variants[plan.variant_id].gpus for plan in prepared.plans]
    return sum(sizes) if prepared.design.preserve_deployment else max(sizes)


def _case_seconds(prepared: Prepared, scenarios: dict[str, dict[str, Any]], case: dict[str, Any]) -> tuple[int, int]:
    estimate = prepared.timing.get(canonical(scenarios[case["scenario"]].get("benchmark") or {})) or {}
    return int(estimate.get("lower_seconds") or 0), int(estimate.get("upper_seconds") or 0)


def eta(prepared: Prepared) -> dict[str, Any]:
    by_plan = {plan.plan_id: {s["id"]: s for s in plan.scenarios} for plan in prepared.plans}
    lower = upper = 0
    for case in prepared.cases:
        low, high = _case_seconds(prepared, by_plan[case["plan"]], case)
        lower, upper = lower + low, upper + high
    variant_of = {plan.plan_id: prepared.variants[plan.variant_id] for plan in prepared.plans}
    deploy = sum(
        DEPLOY_SECONDS
        + (PD_EXTRA_DEPLOY_SECONDS if baseline is None and variant_of[plan_id].fields["guide"] == PD_GUIDE else 0)
        for plan_id, baseline in deployments(prepared)
    )
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


def describe(prepared: Prepared) -> dict[str, Any]:
    peak = peak_gpus(prepared)
    times = eta(prepared)
    runs = deployments(prepared)
    per_deployment = (times["benchmark"]["upper"] + times["deploy"]) / max(1, len(runs))
    by_plan = {plan.plan_id: prepared.variants[plan.variant_id].gpus for plan in prepared.plans}
    gpu_hours = sum(by_plan[plan_id] * per_deployment for plan_id, _ in runs) / 3600
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
                "gpus": prepared.variants[plan.variant_id].gpus,
                "scenarios": [scenario["id"] for scenario in plan.scenarios],
            }
            for plan in prepared.plans
        ],
        "case_order": prepared.cases[:MAX_LISTED_CASES],
        "case_order_truncated": len(prepared.cases) > MAX_LISTED_CASES,
        "cases_total": len(prepared.cases),
        "deployments_total": len(runs),
        "peak_gpus": peak,
        "gpu_hours_estimate": round(gpu_hours, 1),
        "eta_seconds": times,
        "limits": {"plans": MAX_PLANS, "scenarios_per_plan": MAX_SCENARIOS, "baseline_types": MAX_BASELINES},
        "fits_now": None if prepared.free_gpus is None else peak <= prepared.free_gpus,
        "warnings": warnings(prepared),
        "request_digest": prepared.digest,
    }
