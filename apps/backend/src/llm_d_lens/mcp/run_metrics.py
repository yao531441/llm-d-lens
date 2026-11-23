"""Turning Lens benchmark run records into compact results and SLA verdicts."""

from __future__ import annotations

import statistics
from typing import Any

DEFAULT_COMPARE_METRICS = ["ttft_ms.p99", "tpot_ms.p99", "output_tokens_per_s", "success_rate"]
_MAX_POINTS = 60


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantiles(metrics: dict[str, Any], name: str, keys: tuple[str, ...]) -> dict[str, float]:
    distribution = (metrics.get("latency_distributions") or {}).get(name) or {}
    return {key: distribution[f"{key}_ms"] for key in keys if _number(distribution.get(f"{key}_ms"))}


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Key client metrics from a Lens metric summary; success_rate becomes a 0-1 fraction."""
    metrics = metrics or {}
    compact: dict[str, Any] = {
        "ttft_ms": _quantiles(metrics, "ttft", ("p50", "p90", "p95", "p99")),
        "tpot_ms": _quantiles(metrics, "tpot", ("p50", "p90", "p95", "p99")),
        "itl_ms": _quantiles(metrics, "itl", ("p50", "p90", "p99")),
        "e2e_latency_ms": _quantiles(metrics, "request_latency", ("p50", "p90", "p99")),
        "output_tokens_per_s": metrics.get("throughput_tps"),
        "requests_per_s": metrics.get("throughput_rps"),
        "success_rate": round(metrics["success_rate"] / 100, 4) if _number(metrics.get("success_rate")) else None,
    }
    return {key: value for key, value in compact.items() if value not in (None, {})}


def get_metric(metrics: dict[str, Any], path: str) -> float | None:
    node: Any = metrics
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node if _number(node) else None


def run_points(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Every measured point of a run: matrix (isl, osl, concurrency), rate stage, or the single overall summary."""
    points: list[dict[str, Any]] = []
    for item in run.get("matrix_results") or []:
        if item.get("status") == "failed":
            continue
        base = {"isl": item.get("isl"), "osl": item.get("osl")}
        stages = item.get("stage_metrics") or []
        for stage in stages:
            points.append(
                {**base, "concurrency": stage.get("concurrency"), "metrics": compact_metrics(stage.get("metrics"))}
            )
        if not stages:
            points.append({**base, "metrics": compact_metrics(item.get("metrics"))})
    for stage in run.get("rate_stage_results") or []:
        points.append({"rate_rps": stage.get("rate"), "metrics": compact_metrics(stage.get("metrics"))})
    if not points and run.get("metrics"):
        points.append({"metrics": compact_metrics(run["metrics"])})
    return [point for point in points if point["metrics"]][:_MAX_POINTS]


def headline(run: dict[str, Any], points: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Overall summary when Lens has one, else the point with the highest throughput (the saturation point)."""
    if run.get("metrics") and not run.get("matrix_results"):
        return compact_metrics(run["metrics"]), None
    if not points:
        return {}, None
    best = max(points, key=lambda point: point["metrics"].get("output_tokens_per_s") or 0)
    identity = {key: value for key, value in best.items() if key != "metrics"}
    return best["metrics"], identity if len(points) > 1 else None


def unit_result(run: dict[str, Any]) -> dict[str, Any]:
    """Compact, persistable result of one successful Lens benchmark run (one repetition)."""
    points = run_points(run)
    metrics, headline_point = headline(run, points)
    configuration = run.get("configuration") or {}
    return {
        "run_id": run.get("id"),
        "metrics": metrics,
        "headline_point": headline_point,
        "points": points,
        "model": configuration.get("model") or run.get("model"),
        "cluster_id": run.get("cluster_id"),
        "hardware": (run.get("hardware_profile") or {}).get("id"),
        "harness": (run.get("benchmark_runtime") or {}).get("commit"),
        "finished_at": run.get("finished_at"),
        "mechanism_collected": bool((run.get("metrics") or {}).get("mechanism_metrics")),
    }


def _median_nested(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in {key for item in items for key in item}:
        values = [item[key] for item in items if key in item]
        if all(isinstance(value, dict) for value in values):
            merged[key] = _median_nested(values)
        elif numbers := [value for value in values if _number(value)]:
            merged[key] = round(statistics.median(numbers), 4)
    return merged


def sla_verdict(metrics: dict[str, Any], sla: dict[str, Any] | None) -> dict[str, Any]:
    if not sla:
        return {"verdict": "not_set", "checks": []}
    checks = []

    def check(path: str, target: float, actual: float | None, lower_is_better: bool) -> None:
        passed = None if actual is None else (actual <= target if lower_is_better else actual >= target)
        checks.append({"metric": path, "target": target, "actual": actual, "pass": passed})

    if sla.get("ttft_ms") is not None:
        path = f"ttft_ms.{sla.get('ttft_percentile', 'p99')}"
        check(path, sla["ttft_ms"], get_metric(metrics, path), True)
    if sla.get("tpot_ms") is not None:
        path = f"tpot_ms.{sla.get('tpot_percentile', 'p99')}"
        check(path, sla["tpot_ms"], get_metric(metrics, path), True)
    if sla.get("throughput_min_tps") is not None:
        check("output_tokens_per_s", sla["throughput_min_tps"], get_metric(metrics, "output_tokens_per_s"), False)
    if sla.get("success_rate_min_percent") is not None:
        rate = get_metric(metrics, "success_rate")
        check(
            "success_rate_percent",
            sla["success_rate_min_percent"],
            None if rate is None else round(rate * 100, 2),
            False,
        )
    if any(item["pass"] is False for item in checks):
        verdict = "fail"
    elif any(item["pass"] is None for item in checks):
        verdict = "inconclusive"
    else:
        verdict = "pass" if checks else "not_set"
    return {"verdict": verdict, "checks": checks}
