"""Shape Lens simulation tasks for the agent and implement the ``simulation`` resource kind."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .ledger import Resource
from .lens_api import LensApiError

POLL_SECONDS = 2.0
SETTLED = {"completed", "failed", "cancelled"}
ACTIVE = {"queued", "running"}
_STATUS = {"completed": "succeeded"}
_PERCENTILES = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
_CODE_HINTS = {
    400: "The target rejected the request body; check prompt lengths against the deployment's max_model_len.",
    404: "The target did not know the model or route; check that the deployment serves the model name used.",
    408: "Requests timed out waiting for the target; the load may exceed what it can serve.",
    429: "The target or its gateway rate-limited requests; lower scale_factor or add replicas.",
    500: "The model server failed while handling requests; read get_logs for the deployment.",
    502: "The gateway could not reach a backend pod; check the deployment's pods and readiness.",
    503: "The target had no capacity (queue full or pods restarting); lower the load or add replicas.",
    504: "The gateway timed out waiting for the model server; the load likely exceeds capacity.",
}
_NO_RESPONSE_HINT = "No HTTP response was recorded (timeout or connection failure); check the target is still ready."


def public_status(status: str | None) -> str:
    return _STATUS.get(status or "", status or "unknown")


async def fetch_task(ctx: ToolContext, task_id: str) -> dict[str, Any]:
    try:
        return await ctx.api.request("GET", f"/simulation/tasks/{task_id}")
    except LensApiError as error:
        if error.status == 404:
            raise ToolError(
                "not_found", f"Lens has no simulation task '{task_id}'.", hint="It may have been deleted."
            ) from error
        raise


async def wait_for_task(ctx: ToolContext, task_id: str, seconds: float) -> dict[str, Any]:
    """Long-poll until the task settles or ``seconds`` elapse."""
    deadline = asyncio.get_running_loop().time() + seconds
    while True:
        task = await fetch_task(ctx, task_id)
        remaining = deadline - asyncio.get_running_loop().time()
        if task.get("status") in SETTLED or remaining <= 0:
            return task
        await asyncio.sleep(min(POLL_SECONDS, remaining))


def _percentiles(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    return {key: block[key] for key in _PERCENTILES if block.get(key) is not None} or None


def shape_results(result: dict[str, Any] | None) -> dict[str, Any] | None:
    summary = (result or {}).get("summary")
    if not isinstance(summary, dict):
        return None
    shaped: dict[str, Any] = {
        key: summary[key]
        for key in (
            "total_requests",
            "successful_requests",
            "failed_requests",
            "success_rate",
            "duration_seconds",
            "throughput_rps",
            "throughput_tps",
        )
        if summary.get(key) is not None
    }
    if "success_rate" in shaped:
        shaped["success_rate_percent"] = shaped.pop("success_rate")
    for key in ("latency", "ttft", "tpot"):
        block = _percentiles(summary.get(key))
        if block:
            shaped[f"{key}_ms"] = block
    breakdown = summary.get("status_code_breakdown")
    if isinstance(breakdown, list):
        shaped["response_codes"] = [
            {key: item.get(key) for key in ("status_code", "count", "percentage")}
            for item in breakdown[:12]
            if isinstance(item, dict)
        ]
    return shaped


def shape_artifacts(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    artifacts = (result or {}).get("artifacts") or []
    return [
        {
            "kind": item.get("kind"),
            "name": str(item.get("path") or "").rsplit("/", 1)[-1],
            "media_type": item.get("media_type"),
        }
        for item in artifacts
        if isinstance(item, dict)
    ][:20]


def _failing_codes(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    breakdown = ((result or {}).get("summary") or {}).get("status_code_breakdown") or []
    failing = [
        item
        for item in breakdown
        if isinstance(item, dict)
        and not (isinstance(item.get("status_code"), int) and 200 <= item["status_code"] < 300)
    ]
    return sorted(failing, key=lambda item: -(item.get("count") or 0))[:3]


async def response_code_diagnostics(ctx: ToolContext, task_id: str, result: dict[str, Any] | None) -> dict[str, Any]:
    """Top non-2xx codes with a static likely cause and a few sampled requests; error text is untrusted."""
    entries: list[dict[str, Any]] = []
    untrusted: list[str] = []
    for item in _failing_codes(result):
        code = item.get("status_code")
        entry: dict[str, Any] = {
            "status_code": code,
            "count": item.get("count"),
            "percentage": item.get("percentage"),
            "likely_cause": _CODE_HINTS.get(code, _NO_RESPONSE_HINT if not code else f"HTTP {code} from the target."),
        }
        if isinstance(code, int):
            try:
                page = await ctx.api.request(
                    "GET", f"/simulation/tasks/{task_id}/response-code-issues", params={"status_code": code, "limit": 3}
                )
            except LensApiError:
                page = {}
            samples = page.get("issues") or []
            entry["samples"] = [
                {key: sample.get(key) for key in ("request_number", "latency_ms", "input_tokens", "output_tokens")}
                for sample in samples
            ]
            untrusted.extend(str(sample["error"])[:300] for sample in samples if sample.get("error"))
        entries.append(entry)
    diagnostics: dict[str, Any] = {"response_code_issues": entries}
    if untrusted:
        diagnostics["untrusted_text"] = untrusted
    return diagnostics


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    status = task.get("status")
    summary: dict[str, Any] = {
        "status": public_status(status),
        "progress_percent": task.get("progress_percent"),
        "scenario": task.get("scenario"),
        "model": task.get("model_name"),
        "cluster_id": task.get("endpoint_cluster_id"),
        "created_at": task.get("created_at"),
        "completed_at": task.get("completed_at"),
    }
    if task.get("error_message"):
        summary["untrusted_text"] = {"error": str(task["error_message"])[:500]}
    return {key: value for key, value in summary.items() if value is not None}


def estimate_replay(timeline: dict[str, Any], duration_seconds: int, scale_factor: float) -> dict[str, Any]:
    """Requests and peak rate inside the replayed window of the trace (source seconds 0..duration*scale)."""
    window = duration_seconds * scale_factor
    bins = [item for item in timeline.get("bins") or [] if item["start_seconds"] < window]
    requests = 0.0
    peak = 0.0
    for item in bins:
        width = max(item["end_seconds"] - item["start_seconds"], 1e-9)
        covered = min(item["end_seconds"], window) - item["start_seconds"]
        requests += item["request_count"] * covered / width
        peak = max(peak, item["request_count"] / width)
    return {
        "dataset": timeline.get("dataset"),
        "trace_duration_seconds": timeline.get("duration_seconds"),
        "replay_duration_seconds": duration_seconds,
        "scale_factor": scale_factor,
        "estimated_requests": math.ceil(requests),
        "estimated_peak_rps": round(peak * scale_factor, 2),
        "estimated_mean_rps": round(requests / duration_seconds, 2) if duration_seconds else None,
    }


async def delete_simulation(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    task_id = row.lens_execution_id
    if not task_id:
        return
    try:
        task = await fetch_task(ctx, task_id)
    except ToolError:
        return
    if task.get("status") in ACTIVE:
        await ctx.api.request("POST", f"/simulation/tasks/{task_id}/stop")
        task = await wait_for_task(ctx, task_id, 10)
        if task.get("status") in ACTIVE:
            raise ToolError(
                "in_use",
                f"{row.id} is still stopping.",
                hint="Wait a few seconds and delete again; get_simulation shows when it has settled.",
            )
    try:
        await ctx.api.request("DELETE", f"/simulation/tasks/{task_id}")
    except LensApiError as error:
        if error.status == 404:
            return
        if error.status == 409:
            raise ToolError(
                "in_use", f"{row.id} is still running in Lens.", hint="Call cancel_simulation first."
            ) from error
        raise


async def describe_simulation(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    if not row.lens_execution_id:
        return {}
    task = await fetch_task(ctx, row.lens_execution_id)
    return {key: value for key, value in task_summary(task).items() if key != "created_at"}


async def list_other_simulations(ctx: ToolContext, cluster_id: str | None) -> list[dict[str, Any]]:
    page = await ctx.api.request("GET", "/simulation/tasks", params={"limit": 100})
    return [
        {
            "lens_id": task["id"],
            "kind": "simulation",
            "name": task.get("name"),
            "ownership": "unknown",
            "state": public_status(task.get("status")),
            "cluster_id": task.get("endpoint_cluster_id"),
            "created_at": task.get("created_at"),
        }
        for task in page.get("tasks") or []
        if not cluster_id or task.get("endpoint_cluster_id") == cluster_id
    ]
