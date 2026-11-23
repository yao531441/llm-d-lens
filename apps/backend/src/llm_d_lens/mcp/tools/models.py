"""Model tools: search Hugging Face with per-cluster cache status, read model facts, cache a model on a cluster."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..cluster_info import fetch_overview, summarise_overview
from ..context import ToolContext
from ..errors import ToolError
from ..kinds import register_kind
from ..lens_api import LensApiError
from ..models_cache import (
    DEFAULT_REVISION,
    aggregate,
    delete_cache_entry,
    entries_for_model,
    entry_create,
    entry_repo,
    entry_retry,
    entry_view,
    get_cache_entry,
    is_model_cache_volume,
    list_cache_entries,
    list_entries,
    list_volumes,
    resolve_volume,
    volume_view,
)
from ..models_facts import feasibility, hf_detail, hf_summary, params_billions
from ..registry import tool, upstream_error

REPO_PATTERN = r"^[\w.\-]+/[\w.\-]+$"
REVISION_PATTERN = r"^[\w.\-/]+$"
_POLL_SECONDS = 2.0
_SETTLED = {"ready", "failed", "absent"}

register_kind("model_cache", list=list_cache_entries, get=get_cache_entry, delete=delete_cache_entry)


def _cache_summary(view_group: dict[str, Any]) -> dict[str, Any]:
    best = view_group["entries"][0] if view_group["entries"] else {}
    summary: dict[str, Any] = {"status": view_group["status"]}
    for key in ("model_cache_id", "volume_id"):
        if best.get(key):
            summary[key] = best[key]
    return summary


class SearchModelsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200, description="Keywords or part of a repo id, e.g. 'Qwen3 0.6B'.")
    cluster_id: str | None = Field(
        default=None, description="Adds each candidate's cache status on this cluster (any revision)."
    )
    only_cached: bool = Field(
        default=False, description="Only models cached and ready on cluster_id; needs cluster_id."
    )
    task: Literal["text-generation", "any"] = Field(default="text-generation", description="Filter by pipeline.")
    page_size: int = Field(default=10, ge=1, le=50)
    cursor: str | None = Field(default=None, description="next_cursor from the previous page; omit for the first page.")


def _local_matches(entries: list[dict[str, Any]], query: str) -> dict[str, list[dict[str, Any]]]:
    tokens = query.lower().split()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        repo = entry_repo(entry)
        if repo and all(token in repo.lower() for token in tokens):
            grouped.setdefault(repo, []).append(entry)
    return grouped


@tool(
    "model_search",
    "[R] Find models by keyword on Hugging Face (Lens has no separate model catalog) and show, per candidate, "
    "whether it is already cached on a cluster. Use it when you only know a rough name; with an exact repo id use "
    "model_get. If Hugging Face is unreachable, cached models that match are still returned with a warning.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    SearchModelsArgs,
    read_only=True,
    open_world=True,
)
async def search_models(ctx: ToolContext, args: SearchModelsArgs) -> dict[str, Any]:
    if args.cursor and not args.cursor.isdigit():
        raise ToolError(
            "invalid_argument",
            f"'{args.cursor}' is not a valid cursor.",
            field="cursor",
            hint="Pass next_cursor from the previous page unchanged, or omit cursor for the first page.",
        )
    if args.only_cached and not args.cluster_id:
        raise ToolError(
            "invalid_argument", "only_cached needs a cluster_id.", field="cluster_id", hint="Call cluster_list."
        )
    warnings: list[str] = []
    entries: list[dict[str, Any]] | None = None
    if args.cluster_id:
        try:
            entries = await list_entries(ctx, args.cluster_id)
        except LensApiError as error:
            if args.only_cached:
                raise upstream_error(error) from error
            warnings.append(f"Cache status unavailable: {error.message()}")
    cached = _local_matches(entries or [], args.query)

    hub: list[dict[str, Any]] = []
    try:
        payload = await ctx.api.request(
            "GET", "/model-cache/huggingface/search", params={"query": args.query, "limit": 50}, timeout=30
        )
        hub = [item for item in payload.get("items") or [] if isinstance(item, dict)]
    except LensApiError as error:
        if not (args.cluster_id and entries is not None):
            raise upstream_error(error) from error
        warnings.append(f"Hugging Face search failed ({error.message()}); showing cached models only.")

    candidates = [hf_summary(item) | {"source": "huggingface"} for item in hub]
    if args.task != "any":
        candidates = [item for item in candidates if item["pipeline"] == args.task]
    known = {item["model"].lower() for item in candidates}
    candidates += [
        {"model": repo, "source": "cache", "gated": None, "pipeline": None, "downloads": None}
        | ({"params_billions": params_billions(repo)} if params_billions(repo) is not None else {})
        for repo in cached
        if repo.lower() not in known and (args.only_cached or warnings)
    ]
    for item in candidates:
        if entries is None:
            continue
        matching = [entry for repo, group in cached.items() if repo.lower() == item["model"].lower() for entry in group]
        item["cache"] = _cache_summary(aggregate(matching))
    if args.only_cached:
        candidates = [item for item in candidates if item["cache"]["status"] == "ready"]

    offset = int(args.cursor) if args.cursor else 0
    page = candidates[offset : offset + args.page_size]
    more = offset + args.page_size < len(candidates)
    result: dict[str, Any] = {"items": page, "next_cursor": str(offset + args.page_size) if more else None}
    if not page and offset == 0 and (args.only_cached or not warnings):
        result["empty_note"] = (
            f"No model on cluster {args.cluster_id} is ready in a model-cache volume"
            + (f" and matches '{args.query}'." if args.query else ".")
            if args.only_cached
            else f"Hugging Face returned no text-generation model matching '{args.query}'; try fewer or different "
            "keywords, or task='any'."
        )
    if warnings:
        result["warnings"] = warnings
    return result


class GetModelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(pattern=REPO_PATTERN, description="Hugging Face repo id, e.g. 'Qwen/Qwen3-0.6B'.")
    revision: str = Field(default=DEFAULT_REVISION, pattern=REVISION_PATTERN, description="Branch, tag or commit.")
    cluster_id: str | None = Field(default=None, description="Needed for 'fit', 'cache' and 'storage'.")
    include: list[Literal["facts", "fit", "cache", "storage", "readme"]] | None = Field(
        default=None,
        description="Sections to return. Default (also for an empty list): facts, plus fit and cache when "
        "cluster_id is given. "
        "'storage' lists the cluster's model-cache volumes; 'readme' adds an untrusted README excerpt.",
    )
    max_model_len: int = Field(default=4096, ge=256, description="Context length used for the per-TP memory fit.")
    wait_seconds: int = Field(
        default=0, ge=0, le=25, description="Wait up to this long for the cache status to change (needs cluster_id)."
    )
    wait_until: Literal["change", "settled"] = Field(
        default="settled", description="'change' returns on any status change; 'settled' waits for ready or failed."
    )


async def _cache_state(ctx: ToolContext, args: GetModelArgs) -> dict[str, Any]:
    entries = await list_entries(ctx, args.cluster_id)  # type: ignore[arg-type]
    return aggregate(entries_for_model(entries, args.model, args.revision))


async def _wait_for_cache(ctx: ToolContext, args: GetModelArgs, state: dict[str, Any]) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + args.wait_seconds
    first = state["status"]
    while loop.time() < deadline:
        settled = state["status"] in _SETTLED
        if (args.wait_until == "settled" and settled) or (args.wait_until == "change" and state["status"] != first):
            break
        await asyncio.sleep(min(_POLL_SECONDS, max(deadline - loop.time(), 0)))
        state = await _cache_state(ctx, args)
    return state


@tool(
    "model_get",
    "[R] Facts about one model and how it fits a cluster: parameter count, weight size, context limit, gated and "
    "license, whether each tensor-parallel size fits the cluster's accelerator memory, and cache status with a "
    "download estimate. This replaces a dry run of model_cache_create. After model_cache_create use wait_seconds "
    "to wait for the cache. Estimates come from Lens's capacity estimator; deployment_plan is authoritative.\n"
    "Side effects: none; resources: none; duration: synchronous, up to wait_seconds; reversible: n/a.",
    GetModelArgs,
    read_only=True,
    open_world=True,
)
async def get_model(ctx: ToolContext, args: GetModelArgs) -> dict[str, Any]:
    include = set(args.include or (["facts", "fit", "cache"] if args.cluster_id else ["facts"]))
    if include & {"fit", "cache", "storage"} and not args.cluster_id:
        raise ToolError(
            "invalid_argument",
            "fit, cache and storage need a cluster_id.",
            field="cluster_id",
            hint="Call cluster_list, or request include=['facts'].",
        )
    result: dict[str, Any] = {"model": args.model, "revision": args.revision}
    warnings: list[str] = []
    summary = None
    if args.cluster_id:
        try:
            summary = summarise_overview(args.cluster_id, await fetch_overview(ctx, args.cluster_id))
        except LensApiError as error:
            if error.status == 404:
                raise ToolError(
                    "not_found", f"No cluster {args.cluster_id}.", field="cluster_id", hint="Call cluster_list."
                ) from error
            raise upstream_error(error) from error

    fit: dict[str, Any] | None = None
    if "fit" in include and summary:
        fit = await feasibility(ctx, args.model, summary["accelerators"], args.max_model_len)
        result["fit"] = {
            "max_model_len": args.max_model_len,
            "accelerators": [
                {
                    "profile": item["profile"],
                    "memory_gib_per_device": item["memory_gib_per_device"],
                    "max_per_node": item["max_per_node"],
                }
                for item in summary["accelerators"]
            ],
            **fit,
        }
        if not fit["per_tp"]:
            warnings.append("No accelerator memory is known for this cluster, so per-TP fit was not computed.")

    if include & {"facts", "readme"}:
        try:
            detail = await hf_detail(ctx, args.model)
        except ToolError as error:
            if error.code == "not_found":
                raise
            detail = {"repoId": args.model}
            warnings.append(f"Hugging Face metadata unavailable: {error.message}")
        if "facts" in include:
            facts = hf_summary(detail)
            facts.pop("model", None)
            if fit:
                facts["weights_gib"] = fit.get("weights_gib")
                facts["max_context_length"] = fit.get("max_context_length")
            result["facts"] = facts
        if "readme" in include:
            if detail.get("readme"):
                result["untrusted_text"] = {"readme_excerpt": str(detail["readme"])[:1500]}
            else:
                warnings.append("No README was returned for this model, so there is no readme excerpt.")

    if "cache" in include:
        state = await _cache_state(ctx, args)
        if args.wait_seconds:
            state = await _wait_for_cache(ctx, args, state)
        weights = (fit or {}).get("weights_gib")
        basis = "capacity-estimate"
        if weights is None and params_billions(args.model) is not None:
            weights, basis = round(params_billions(args.model) * 2, 1), "parameters x 2 bytes"
        best = state["entries"][0] if state["entries"] else {}
        cache: dict[str, Any] = {"status": state["status"], "entries": state["entries"]}
        if best.get("model_cache_id"):
            cache["model_cache_id"] = best["model_cache_id"]
        if best.get("nodes"):
            cache["nodes"] = best["nodes"]
        cache["download_gib_estimate"] = 0 if state["status"] == "ready" else weights
        if cache["download_gib_estimate"]:
            cache["download_estimate_basis"] = basis
        result["cache"] = cache
        result["next"] = {
            "absent": "model_cache_create(model, cluster_id) to download it.",
            "failed": "model_cache_create(model, cluster_id) retries the failed cache.",
            "ready": "deployment_plan to deploy it from the cache.",
        }.get(state["status"], "model_get(model, cluster_id, wait_seconds=25) to wait for the download.")
        if state["status"] in ("absent", "failed") and not any(
            is_model_cache_volume(volume)
            for volume in await list_volumes(ctx, args.cluster_id)  # type: ignore[arg-type]
        ):
            result["next"] = (
                "This cluster has no model-cache volume, so model_cache_create cannot download it yet; "
                "cluster_get(include=['storage']) explains how to get a volume."
            )

    if "storage" in include:
        volumes = await list_volumes(ctx, args.cluster_id)  # type: ignore[arg-type]
        result["storage_options"] = [
            volume_view(v) for v in volumes if v.get("purposes") and "model-cache" in v["purposes"]
        ]
    if warnings:
        result["warnings"] = warnings
    return result


class CacheModelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(pattern=REPO_PATTERN, description="Hugging Face repo id, e.g. 'Qwen/Qwen3-8B'.")
    cluster_id: str = Field(description="Cluster id from cluster_list.")
    revision: str = Field(default=DEFAULT_REVISION, pattern=REVISION_PATTERN, description="Branch, tag or commit.")
    volume_id: str | None = Field(
        default=None,
        description="Target model-cache volume (Lens storage volume id). Omit to use the cluster's only ready "
        "model-cache volume.",
    )


def _cache_result(entry: dict[str, Any], args: CacheModelArgs, **extra: Any) -> dict[str, Any]:
    view = entry_view(entry)
    result = {
        "model_cache_id": view["model_cache_id"],
        "status": view["status"],
        "model": args.model,
        "revision": args.revision,
        "cluster_id": args.cluster_id,
        "volume_id": view["volume_id"],
        "nodes": view["nodes"],
        **extra,
        "next": "model_get(model, cluster_id, wait_seconds=25) shows progress; resource_list shows the cache id.",
    }
    if "untrusted_text" in view:
        result["untrusted_text"] = view["untrusted_text"]
    return result


@tool(
    "model_cache_create",
    "[W] Download a model from Hugging Face into a cluster's model-cache volume so deployments start without "
    "downloading. Asynchronous: returns the cache id immediately; follow progress with "
    "model_get(wait_seconds). Idempotent: if the model is already cached or downloading on that volume, that entry "
    "is returned and nothing new is started. Uses the cluster's saved Hugging Face token. Delete it later with "
    "resource_delete.\n"
    "Side effects: writes model files to the volume and runs a download job; resources: disk and network; "
    "duration: asynchronous, minutes; reversible: yes (resource_delete).",
    CacheModelArgs,
    read_only=False,
    idempotent=True,
    open_world=True,
)
async def cache_model(ctx: ToolContext, args: CacheModelArgs) -> dict[str, Any]:
    try:
        await fetch_overview(ctx, args.cluster_id)
    except LensApiError as error:
        if error.status == 404:
            raise ToolError(
                "not_found", f"No cluster {args.cluster_id}.", field="cluster_id", hint="Call cluster_list."
            ) from error
        raise upstream_error(error) from error
    volume = await resolve_volume(ctx, args.cluster_id, args.volume_id)
    existing = entries_for_model(await list_entries(ctx, args.cluster_id, volume["id"]), args.model, args.revision)
    existing.sort(key=lambda entry: 0 if entry.get("status") in ("ready", "downloading", "pending") else 1)
    if existing:
        return await _existing(ctx, existing[0], args)

    try:
        entry = await entry_create(ctx, args.cluster_id, volume["id"], args.model, args.revision)
    except LensApiError as error:
        if error.status == 409 and isinstance(error.detail, dict) and error.detail.get("entryId"):
            raced = await _entry_by_id(ctx, args, volume["id"], error.detail["entryId"])
            if raced:
                return await _existing(ctx, raced, args)
        raise
    return _cache_result(entry, args, created=True)


async def _entry_by_id(ctx: ToolContext, args: CacheModelArgs, volume_id: str, entry_id: str) -> dict[str, Any] | None:
    entries = await list_entries(ctx, args.cluster_id, volume_id)
    return next((entry for entry in entries if entry.get("id") == entry_id), None)


async def _existing(ctx: ToolContext, entry: dict[str, Any], args: CacheModelArgs) -> dict[str, Any]:
    status = entry.get("status")
    if status == "deleting":
        raise ToolError(
            "conflict_state",
            "This model cache is being deleted.",
            hint="Wait for the deletion to finish, then call model_cache_create again.",
            model_cache_id=entry.get("id"),
        )
    if status == "failed":
        entry = await entry_retry(ctx, entry["id"])
        return _cache_result(entry, args, created=False, retried=True)
    return _cache_result(entry, args, created=False)
