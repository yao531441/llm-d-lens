"""plan_deployment pipeline: cluster session -> guide plan -> render. Pure computation, no cluster changes."""

from __future__ import annotations

import re
import uuid
from typing import Any

import yaml

from .cluster_info import cluster_summary
from .context import ToolContext
from .errors import ToolError

_POD_TEMPLATE_PATHS = (("spec", "template"), ("spec", "leaderWorkerTemplate", "workerTemplate"))


def slug(value: str, limit: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text[:limit].strip("-") or "model"


def serving_image(manifest: str, workload_names: list[str]) -> str | None:
    """Image of the model-server container, which render requires the candidate config to repeat exactly."""
    for document in yaml.safe_load_all(manifest):
        if not isinstance(document, dict) or (document.get("metadata") or {}).get("name") not in workload_names:
            continue
        for path in _POD_TEMPLATE_PATHS:
            node: Any = document
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            containers = ((node or {}).get("spec") or {}).get("containers") or []
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
            hint="Call list_guides and choose one of the returned guide ids.",
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


def planning_request(args, *, accelerator: str, server: str, variant: str, session_id: str) -> dict[str, Any]:
    is_pd = args.guide == "pd-disaggregation"
    request: dict[str, Any] = {
        "guide": args.guide,
        "accelerator": accelerator,
        "modelServer": server,
        "guideVariant": variant,
        "model": args.model,
        "modelSource": "huggingface",
        "replicas": args.replicas,
        "tensorParallelSize": args.tensor_parallel_size,
        "decodeReplicas": args.replicas,
        "decodeTensorParallelSize": args.tensor_parallel_size,
        "prefillReplicas": args.prefill_replicas if is_pd else 0,
        "prefillTensorParallelSize": args.prefill_tensor_parallel_size if is_pd else 0,
        "clusterSessionId": session_id,
        "source": {"mode": "official"},
        "guideSettings": {},
    }
    if server == "vllm" and args.max_model_len:
        request["maxModelLen"] = args.max_model_len
    return request


def candidate_config(args, *, cluster: dict[str, Any], server: str, variant: str, image: str) -> dict[str, Any]:
    is_pd = args.guide == "pd-disaggregation"
    serving = {"replicas": args.replicas, "tensor_parallel_size": args.tensor_parallel_size}
    decode = dict(serving)
    if server == "vllm" and args.max_model_len:
        decode["max_model_len"] = args.max_model_len
    return {
        "schema_version": "1.0",
        "type": "pd" if is_pd else "baseline",
        "guide_settings": {},
        "candidate_source": {"name": "manual"},
        "target": {"model": args.model, "cluster_id": cluster["id"], "cluster_name": cluster["name"]},
        "serving": serving,
        "prefill": (
            {"replicas": args.prefill_replicas, "tensor_parallel_size": args.prefill_tensor_parallel_size}
            if is_pd
            else None
        ),
        "decode": decode,
        "custom_parameters": [],
        "guide_variant": variant,
        "runtime": {
            "image": image,
            "imageMode": "use-upstream-image",
            "buildSourceUrl": "",
            "modelServer": server,
            "resourceBasis": "available",
            "modelPvcClaimName": "",
            "modelSource": "huggingface",
            "mountPath": "",
            "storageVolumeId": "",
        },
    }


async def build_plan(ctx: ToolContext, args) -> dict[str, Any]:
    summary = await cluster_summary(ctx, args.cluster_id)
    accelerator = pick_accelerator(args.accelerator, summary)
    session = await ctx.api.request("GET", "/cluster/session", params={"clusterId": args.cluster_id})
    session_id, cluster = session["sessionId"], session["cluster"]
    catalog = await ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": args.cluster_id})
    server, variant = resolve_guide(catalog, args.guide, accelerator, args.model_server, args.variant)

    plan = await ctx.api.request(
        "POST",
        "/guide-planning/plan",
        body=planning_request(args, accelerator=accelerator, server=server, variant=variant, session_id=session_id),
        timeout=240,
    )
    validation = plan.get("validation") or {}
    if plan.get("status") not in ("VERIFIED", "PLANNED") or validation.get("errors"):
        raise ToolError(
            "plan_invalid",
            "; ".join(validation.get("errors") or []) or f"Guide planning returned status {plan.get('status')}.",
            hint=(
                "Reduce replicas or tensor_parallel_size, pick another guide, "
                "or check get_cluster for free accelerators."
            ),
            warnings=validation.get("warnings") or [],
        )
    manifest = (plan.get("plannedDeployment") or {}).get("content")
    if not manifest:
        raise ToolError(
            "plan_empty",
            "The guide produced an empty manifest.",
            hint="Retry; if it persists the guide ref may be broken.",
        )

    workloads = (plan.get("deployment") or {}).get("readinessDeployments") or []
    image = args.image or serving_image(manifest, workloads)
    if not image:
        raise ToolError(
            "image_unresolved",
            "Could not determine the model-server image from the guide manifest.",
            field="image",
            hint="Pass image explicitly.",
        )

    name = args.name or f"mcp-{slug(args.model.rsplit('/', 1)[-1], 30)}-{uuid.uuid4().hex[:4]}"
    rendered = await ctx.api.request(
        "POST",
        "/configurations/render",
        body={
            "candidate_config": candidate_config(args, cluster=cluster, server=server, variant=variant, image=image),
            "render": {
                "format": "manifest",
                "guide_ref": args.guide,
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

    plan_summary = plan.get("summary") or {}
    description = {
        "guide": args.guide,
        "accelerator": accelerator,
        "model_server": server,
        "variant": variant,
        "model": args.model,
        "deployment_name": name,
        "image": image,
        "replicas": args.replicas,
        "tensor_parallel_size": args.tensor_parallel_size,
        "accelerators_per_replica": plan_summary.get("acceleratorsPerReplica"),
        "accelerators_free": plan_summary.get("availableAccelerators"),
        "validation": {"status": validation.get("status"), "warnings": validation.get("warnings") or []},
    }
    plan_id = ctx.ledger.put_plan(
        owner=ctx.owner,
        cluster_id=args.cluster_id,
        payload={
            "cluster_id": cluster["id"],
            "cluster_session_id": session_id,
            "deployable_configuration": rendered["deployable_configuration"],
            "description": description,
        },
    )
    return {
        "plan_id": plan_id,
        **description,
        "expires_in_seconds": 3600,
        "next": "Call create_deployment with this plan_id.",
    }
