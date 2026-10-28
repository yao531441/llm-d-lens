"""plan_deployment(mode='recommend'): rank guide / replica / tensor-parallel choices. Heuristic, read-only."""

from __future__ import annotations

from typing import Any

from .cluster_info import cluster_summary
from .context import ToolContext
from .deploy_spec import PD_GUIDE, DeploymentSpec
from .errors import ToolError
from .lens_api import LensApiError
from .planning import pick_accelerator

GIB = 1024**3
TP_CANDIDATES = (1, 2, 4, 8)
MAX_CANDIDATES = 5
RATIONALE = {
    PD_GUIDE: "Separates prefill from decode; helps long prompts, needs at least two pods.",
    "optimized-baseline": "Single serving role behind the llm-d router; the simplest, cheapest start.",
}


async def _per_gpu_gib(ctx: ToolContext, cluster_id: str) -> float | None:
    overview = await ctx.api.request("GET", "/cluster/overview", params={"clusterId": cluster_id})
    hardware = ((overview.get("kubernetes") or {}).get("hardware")) or {}
    if hardware.get("vramBytes") and hardware.get("gpuCount"):
        return hardware["vramBytes"] / hardware["gpuCount"] / GIB
    return None


async def _smallest_tp(ctx: ToolContext, spec: DeploymentSpec, gpu_gib: float | None, free: int) -> tuple[int, str]:
    """Smallest tensor-parallel size the Lens capacity estimator accepts, and how that was decided."""
    if gpu_gib is None:
        return 1, "unchecked: per-accelerator memory unknown"
    reasons: list[str] = []
    for tp in (size for size in TP_CANDIDATES if size <= max(free, 1)):
        try:
            estimate = await ctx.api.request(
                "POST",
                "/capacity/estimate",
                body={
                    "model": spec.model,
                    "gpu_memory_gib": round(gpu_gib, 2),
                    "tensor_parallel_size": tp,
                    "max_model_len": spec.engine.max_model_len or 4096,
                },
                timeout=60,
            )
        except LensApiError:
            return 1, "unchecked: Lens capacity estimator unavailable (model config not readable)"
        if estimate.get("is_deployable"):
            return tp, f"capacity estimator accepts TP={tp} on {gpu_gib:.0f} GiB accelerators"
        reasons = estimate.get("rejection_reasons") or reasons
    raise ToolError(
        "insufficient_capacity",
        f"No tensor-parallel size up to {max(free, 1)} fits {spec.model} on this cluster's accelerators.",
        field="spec.model",
        hint="Choose a smaller model or free accelerators (list_resources, stop_deployment).",
        reasons=reasons[:3],
    )


async def recommend(ctx: ToolContext, spec: DeploymentSpec) -> dict[str, Any]:
    summary = await cluster_summary(ctx, spec.cluster_id)
    accelerator = pick_accelerator(spec.accelerator, summary)
    free = summary.get("accelerators_free") or 0
    catalog = await ctx.api.request("GET", "/guide-planning/catalog", params={"clusterId": spec.cluster_id})
    server = spec.engine_kind or "vllm"
    guides = [
        entry["id"]
        for entry in catalog.get("guides") or []
        if (spec.guide is None or entry["id"] == spec.guide)
        and any(
            accel["id"] == accelerator and any(ms["id"] == server for ms in accel.get("modelServers") or [])
            for accel in entry.get("accelerators") or []
        )
    ]
    if not guides:
        raise ToolError(
            "unsupported_combination",
            f"No guide offers {server} on {accelerator}.",
            field="spec.guide",
            hint="Call list_guides to see what is available.",
        )
    tp, basis = await _smallest_tp(ctx, spec, await _per_gpu_gib(ctx, spec.cluster_id), free)
    candidates = []
    for guide in guides:
        topology = (
            {
                "prefill": {"replicas": 1, "tensor_parallel_size": tp},
                "decode": {"replicas": 1, "tensor_parallel_size": tp},
            }
            if guide == PD_GUIDE
            else {"serving": {"replicas": 1, "tensor_parallel_size": tp}}
        )
        total = tp * (2 if guide == PD_GUIDE else 1)
        candidates.append(
            {
                "spec": {
                    "cluster_id": spec.cluster_id,
                    "model": spec.model,
                    "guide": guide,
                    "accelerator": accelerator,
                    "engine_kind": server,
                    "topology": topology,
                },
                "gpus_total": total,
                "fits_now": total <= free,
                "rationale": RATIONALE.get(guide, f"Guide '{guide}' supports {accelerator}/{server}."),
            }
        )
    candidates.sort(key=lambda item: (not item["fits_now"], item["gpus_total"], item["spec"]["guide"]))
    ranked = [{"rank": index + 1, **item} for index, item in enumerate(candidates[:MAX_CANDIDATES])]
    return {
        "mode": "recommend",
        "heuristic": True,
        "basis": f"Ranked by fit on {free} free accelerators, then by cost; {basis}. Lens has no performance model "
        "behind this ranking, so it says nothing about latency or throughput.",
        "candidates": ranked,
        "next": "Call plan_deployment(mode='validate', spec=<candidate spec>) to validate one candidate.",
    }
