"""Plan pipeline: cluster session -> guide plan -> render. Pure computation, no cluster changes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from .context import ToolContext
from .errors import ToolError
from .guide_fields import LONE_DEPLOYMENT, gpu_count, plan_request, reject_misplaced_pd_fields
from .guide_fields import candidate_config as guide_candidate_config
from .models_cache import (
    entries_for_model,
    is_model_cache_volume,
    list_entries,
    list_volumes,
    no_usable_volume,
    resolve_volume,
)

_POD_TEMPLATE_PATHS = (("spec", "template"), ("spec", "leaderWorkerTemplate", "workerTemplate"))


# What a plan does not verify; plans only render and validate, they never touch the cluster's nodes or registry.
NOT_CHECKED = [
    "Whether the image can be pulled by the cluster's nodes.",
    "Whether the model files exist at an explicit modelPath (checked only for models from a registered cache).",
    "That the accelerators stay free until the deployment or experiment is actually created.",
]


def invalid_plan(error: ToolError, next_step: str) -> dict[str, Any]:
    """A plan result for a problem with the proposed deployment.

    Malformed or contradictory call arguments (``invalid_argument``) and failures of Lens or the cluster stay
    tool errors.
    """
    if error.code == "invalid_argument" or error.extra.get("upstream_status") is not None:
        raise error
    problem = {
        key: value
        for key, value in {
            "code": error.code,
            "message": error.message,
            "field": error.field,
            "hint": error.hint,
        }.items()
        if value
    }
    if "choices" in error.extra:
        problem["choices"] = error.extra["choices"]
    return {
        "status": "invalid",
        "validation": {"errors": [problem], "warnings": error.extra.get("warnings") or []},
        "next": next_step,
    }


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
    source: dict[str, Any] = field(default_factory=dict)
    applied_patches: list[dict[str, Any]] = field(default_factory=list)


async def plan_configuration(
    ctx: ToolContext,
    variant_id: str,
    fields: dict[str, Any],
    *,
    cluster_id: str,
    cluster: dict[str, Any],
    session_id: str,
    summary: dict[str, Any],
    catalog: dict[str, Any],
    deployment_name: str,
    model_cache_volume_id: str | None = None,
) -> VariantPlan:
    """Plan and render shared guide fields through the same Lens APIs for every caller."""
    accelerator = pick_accelerator(fields.get("accelerator"), summary)
    source_mode = (fields.get("source") or {}).get("mode", "official")
    if source_mode == "official":
        try:
            server, guide_variant = resolve_guide(
                catalog, fields["guide"], accelerator, fields.get("modelServer"), fields.get("guideVariant")
            )
        except ToolError as error:
            raise _in_variant(error, variant_id) from error
    else:
        # Local/remote manifests are selected by Lens' source loader, not the pinned official catalog.
        server, guide_variant = fields.get("modelServer") or "vllm", fields.get("guideVariant") or "base"
    reject_misplaced_pd_fields(fields, None if variant_id == LONE_DEPLOYMENT else variant_id)
    if model_cache_volume_id and any(fields.get(key) for key in ("modelSource", "modelPath", "modelPvcClaimName")):
        raise ToolError(
            "field_conflict",
            "model_cache_volume_id cannot be combined with explicit model storage fields.",
            field="model_cache_volume_id",
        )
    if not fields.get("modelSource") and not fields.get("modelPath"):
        volume = await cached_volume(ctx, cluster_id, fields["model"], model_cache_volume_id)
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
            hint="Adjust the configuration fields; Lens validation messages are quoted above.",
        )
    manifest = (plan.get("plannedDeployment") or {}).get("content")
    if not manifest:
        raise ToolError("plan_empty", f"Variant '{variant_id}': the guide produced an empty manifest.")
    guide_variant = plan["source"].get("variant") or guide_variant
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
    rendered = await ctx.api.request(
        "POST",
        "/configurations/render",
        body={
            "candidate_config": guide_candidate_config(
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
                "deploymentName": deployment_name,
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
        plan["source"],
        list(plan.get("appliedCustomPatches") or []),
    )


def slug(value: str, limit: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text[:limit].strip("-") or "model"


def _pod_templates(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Pod templates of a workload; a DisaggregatedSet carries one per role (decode first)."""
    templates: list[Any] = []
    for path in _POD_TEMPLATE_PATHS:
        node: Any = document
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        templates.append(node)
    roles = sorted((document.get("spec") or {}).get("roles") or [], key=lambda role: role.get("name") != "decode")
    for role in roles:
        templates.append((((role.get("spec") or {}).get("leaderWorkerTemplate")) or {}).get("workerTemplate"))
    return [template for template in templates if isinstance(template, dict)]


def serving_image(manifest: str, workload_names: list[str]) -> str | None:
    """Image of the model-server container, which render requires the candidate config to repeat exactly."""
    for document in yaml.safe_load_all(manifest):
        if not isinstance(document, dict) or (document.get("metadata") or {}).get("name") not in workload_names:
            continue
        for template in _pod_templates(document):
            containers = (template.get("spec") or {}).get("containers") or []
            if containers:
                chosen = next((c for c in containers if c.get("name") == "modelserver"), containers[0])
                return chosen.get("image")
    return None


def pick_accelerator(requested: str | None, summary: dict[str, Any]) -> str:
    if requested:
        return requested
    families = sorted({item["family"] for item in summary["accelerators"]})
    if len(families) == 1:
        return families[0]
    raise ToolError(
        "accelerator_ambiguous" if families else "no_accelerators",
        "Cannot infer the accelerator family for this cluster." if families else "The cluster reports no accelerators.",
        field="accelerator",
        hint=f"Pass accelerator explicitly; cluster offers: {', '.join(families) or 'none'}.",
    )


def resolve_guide(catalog: dict[str, Any], guide: str, accelerator: str, server: str | None, variant: str | None):
    entry = next((g for g in catalog.get("guides") or [] if g["id"] == guide), None)
    if entry is None:
        raise ToolError(
            "unknown_guide",
            f"Guide '{guide}' is not in the catalog.",
            field="guide",
            hint="Call guide_list and choose one of the returned guide ids.",
        )
    accel = next((a for a in entry.get("accelerators") or [] if a["id"] == accelerator), None)
    if accel is None:
        offered = [a["id"] for a in entry.get("accelerators") or []]
        raise ToolError(
            "unsupported_accelerator",
            f"Guide '{guide}' does not support accelerator '{accelerator}'.",
            field="accelerator",
            hint=f"This guide supports: {', '.join(offered)}.",
        )
    servers = {s["id"]: s.get("variants") or [] for s in accel.get("modelServers") or []}
    server = server or ("vllm" if "vllm" in servers else next(iter(servers), None))
    if server not in servers:
        raise ToolError(
            "unsupported_model_server",
            f"Model server '{server}' is not available for {guide}/{accelerator}.",
            field="model_server",
            hint=f"Available: {', '.join(servers)}.",
        )
    variants = servers[server]
    variant = variant or ("base" if "base" in variants or not variants else variants[0])
    if variants and variant not in variants:
        raise ToolError(
            "unsupported_variant",
            f"Variant '{variant}' is not available for {guide}/{accelerator}/{server}.",
            field="variant",
            hint=f"Available: {', '.join(variants)}.",
        )
    return server, variant


_PLAN_FIELD_NAMES = {"model_server": "modelServer", "variant": "guideVariant"}


def _in_variant(error: ToolError, variant_id: str) -> ToolError:
    """Name the experiment variant a guide-resolution error belongs to; a lone deployment keeps the bare error."""
    if variant_id == LONE_DEPLOYMENT:
        return error
    field = _PLAN_FIELD_NAMES.get(error.field or "", error.field or "guide")
    return ToolError(
        error.code,
        f"Variant '{variant_id}': {error.message}",
        field=f"variants/{variant_id}/plan/{field}",
        hint=error.hint,
        **error.extra,
    )


def location_caveat(next_step: str, resolved: list[dict[str, Any]]) -> str:
    """Qualify a plan's next step when a model location was given explicitly, because Lens cannot verify it."""
    if all(fields.get("modelSource") == "auto-cache" for fields in resolved):
        return next_step
    return (
        "The model location was given explicitly and is not verified (see not_checked). If the path is only a "
        "placeholder this is a preview and not deployable; otherwise " + next_step[0].lower() + next_step[1:]
    )


PREVIEW_WITHOUT_CACHE = (
    " To preview the plan before a volume exists, set modelPath (an absolute path on the worker nodes): "
    "configuration.modelPath for deployment_plan, common.modelPath or variants[].plan.modelPath for "
    "experiment_plan; modelSource then defaults to shared-path. Advanced: Lens does not check that the path exists, "
    "so the result is not a deployable plan."
)


async def cached_volume(ctx: ToolContext, cluster_id: str, model: str, requested: str | None) -> dict[str, Any]:
    """The model-cache volume holding a ready copy of ``model``; Lens deployments only start from a cache."""
    entries = [
        entry
        for entry in entries_for_model(await list_entries(ctx, cluster_id), model, None)
        if entry.get("status") == "ready"
    ]
    if requested:
        volume = await resolve_volume(ctx, cluster_id, requested, "model_cache_volume_id")
        if not any(entry.get("storageVolumeId") == volume["id"] for entry in entries):
            raise ToolError(
                "model_not_cached",
                f"{model} is not ready in volume {requested}.",
                field="model_cache_volume_id",
                hint=f"Call model_cache_create(model='{model}', volume_id=...) and wait until it is ready.",
            )
        return volume
    volumes = sorted({entry["storageVolumeId"] for entry in entries if entry.get("storageVolumeId")})
    if len(volumes) == 1:
        return await resolve_volume(ctx, cluster_id, volumes[0])
    if not volumes:
        cluster_volumes = await list_volumes(ctx, cluster_id)
        if not any(is_model_cache_volume(volume) and volume.get("status") == "ready" for volume in cluster_volumes):
            raise no_usable_volume(ctx, cluster_id, cluster_volumes, PREVIEW_WITHOUT_CACHE, "model_cache_volume_id")
        raise ToolError(
            "model_not_cached",
            f"{model} is not downloaded into any model-cache volume on cluster {cluster_id}.",
            field="configuration.model",
            hint=(
                f"Call model_cache_create(cluster_id='{cluster_id}', model='{model}'), "
                "wait until ready, then plan again." + PREVIEW_WITHOUT_CACHE
            ),
        )
    raise ToolError(
        "ambiguous_choice",
        f"{model} is cached in {len(volumes)} volumes; choose one.",
        field="model_cache_volume_id",
        choices=volumes,
    )
