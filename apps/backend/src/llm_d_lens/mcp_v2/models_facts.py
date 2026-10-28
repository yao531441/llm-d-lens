"""Model facts for ``get_model`` and ``search_models``: Hugging Face metadata and per-TP memory feasibility.

Lens has no model catalog or registry. Facts come from the Hugging Face proxy and the read-only capacity estimator
(``POST /capacity/estimate`` computes only; it creates nothing).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .lens_api import LensApiError
from .registry import upstream_error

_PARAMS_PATTERN = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)
_MAX_CONTEXT_PATTERN = re.compile(r"exceeds model limit of (\d+)")
_PROBE_CONTEXT = 100_000_000
_TP_CANDIDATES = (1, 2, 4, 8)


def params_billions(repo_id: str) -> float | None:
    """Parameter count read from the repository name (``Llama-3.1-8B-Instruct`` -> 8.0); None when it has none."""
    match = _PARAMS_PATTERN.search(repo_id.rsplit("/", 1)[-1])
    return float(match.group(1)) if match else None


def hf_summary(item: dict[str, Any]) -> dict[str, Any]:
    tags = item.get("tags") or []
    licenses = [tag.split(":", 1)[1] for tag in tags if isinstance(tag, str) and tag.startswith("license:")]
    summary = {
        "model": item.get("repoId"),
        "gated": bool(item.get("gated")),
        "pipeline": item.get("pipelineTag"),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "license": licenses[0] if licenses else None,
        "last_modified": item.get("lastModified"),
    }
    billions = params_billions(item.get("repoId") or "")
    if billions is not None:
        summary["params_billions"] = billions
    return summary


async def hf_detail(ctx: ToolContext, model: str) -> dict[str, Any]:
    try:
        return await ctx.api.request("GET", f"/model-cache/huggingface/models/{model}", timeout=30)
    except LensApiError as error:
        if error.status == 404:
            raise ToolError(
                "not_found",
                f"Model '{model}' was not found on Hugging Face.",
                field="model",
                hint="Use search_models to find the exact repository id.",
            ) from error
        raise upstream_error(error) from error


async def estimate(
    ctx: ToolContext, model: str, *, gpu_memory_gib: float, tensor_parallel_size: int, max_model_len: int
) -> dict[str, Any]:
    return await ctx.api.request(
        "POST",
        "/capacity/estimate",
        body={
            "model": model,
            "gpu_memory_gib": gpu_memory_gib,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
        },
        timeout=60,
    )


def _fit(accelerator: dict[str, Any], tp: int, result: dict[str, Any]) -> dict[str, Any]:
    weights = (result.get("memory_breakdown") or {}).get("model_weights_gib")
    return {
        "accelerator": accelerator["profile"],
        "tensor_parallel_size": tp,
        "fits": bool(result.get("is_deployable")),
        "weights_gib_per_device": round(weights / tp, 2) if weights else None,
        "kv_cache_gib_per_device": result.get("allocatable_kv_cache_gib"),
        "max_concurrent_requests": result.get("max_concurrent_requests"),
        "reasons": [str(reason)[:300] for reason in result.get("rejection_reasons") or []],
    }


async def _fit_one(
    ctx: ToolContext, model: str, accelerator: dict[str, Any], tp: int, max_model_len: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        result = await estimate(
            ctx,
            model,
            gpu_memory_gib=accelerator["memory_gib_per_device"],
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
        )
    except LensApiError as error:
        return {
            "accelerator": accelerator["profile"],
            "tensor_parallel_size": tp,
            "fits": None,
            "error": error.message(),
        }, None
    return _fit(accelerator, tp, result), result


async def feasibility(
    ctx: ToolContext, model: str, accelerators: list[dict[str, Any]], max_model_len: int
) -> dict[str, Any]:
    """Memory fit of ``model`` per accelerator profile and tensor-parallel size, plus size and context facts."""
    usable = [item for item in accelerators if item.get("memory_gib_per_device")]
    jobs = [
        _fit_one(ctx, model, item, tp, max_model_len)
        for item in usable
        for tp in _TP_CANDIDATES
        if tp <= (item.get("max_per_node") or max(_TP_CANDIDATES))
    ]
    if not jobs:
        return {"per_tp": [], "method": "unavailable"}
    outcomes, max_context = await asyncio.gather(
        asyncio.gather(*jobs), _max_context(ctx, model, usable[0]["memory_gib_per_device"])
    )
    results = [result for _, result in outcomes if result]
    facts: dict[str, Any] = {"per_tp": [fit for fit, _ in outcomes]}
    if results:
        first = results[0]
        facts["weights_gib"] = (first.get("memory_breakdown") or {}).get("model_weights_gib")
        facts["valid_tensor_parallel_sizes"] = first.get("possible_tps") or []
        facts["method"] = "model-config" if first.get("possible_tps") else "name-heuristic"
    if max_context:
        facts["max_context_length"] = max_context
    return facts


async def _max_context(ctx: ToolContext, model: str, gpu_memory_gib: float) -> int | None:
    """The model's context limit, read from the estimator's rejection of an absurdly long context."""
    try:
        result = await estimate(
            ctx, model, gpu_memory_gib=gpu_memory_gib, tensor_parallel_size=1, max_model_len=_PROBE_CONTEXT
        )
    except LensApiError:
        return None
    for text in result.get("rejection_reasons") or []:
        match = _MAX_CONTEXT_PATTERN.search(str(text))
        if match:
            return int(match.group(1))
    return None
