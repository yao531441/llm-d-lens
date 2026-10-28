"""Turning Lens benchmark run records into compact results, SLA verdicts and comparisons."""

from __future__ import annotations

import statistics
from typing import Any

DEFAULT_COMPARE_METRICS = ["ttft_ms.p99", "tpot_ms.p99", "output_tokens_per_s", "success_rate"]
_LOWER_IS_BETTER = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_latency_ms")
_PER_GPU = {"output_tokens_per_s", "requests_per_s"}
_TIME_GAP_SECONDS = 7 * 24 * 3600
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


def aggregate(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Median over repetitions of the same (variant, workload); records how noisy the repetitions were."""
    metrics = _median_nested([unit["metrics"] for unit in units])
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for unit in units:
        for point in unit["points"]:
            identity = tuple(sorted((k, v) for k, v in point.items() if k != "metrics"))
            grouped.setdefault(identity, []).append(point["metrics"])
    points = [{**dict(identity), "metrics": _median_nested(values)} for identity, values in grouped.items()]
    result = {**units[0], "metrics": metrics, "points": points, "repetitions": len(units)}
    throughput = [
        u["metrics"].get("output_tokens_per_s") for u in units if _number(u["metrics"].get("output_tokens_per_s"))
    ]
    if len(throughput) > 1 and statistics.median(throughput):
        result["throughput_spread_pct"] = round(
            (max(throughput) - min(throughput)) / statistics.median(throughput) * 100, 1
        )
    return result


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


def _delta(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base == 0:
        return None
    return round((value - base) / base, 4)


def _normalized(entry: dict[str, Any], path: str, per_gpu: bool) -> float | None:
    value = get_metric(entry["metrics"], path)
    gpus = entry.get("gpus")
    if per_gpu and path in _PER_GPU and value is not None and gpus:
        return round(value / gpus, 4)
    return value


def compare_entries(
    entries: list[dict[str, Any]], baseline_ref: str, metrics: list[str], per_gpu: bool = False
) -> dict[str, Any]:
    """Rows of values and relative deltas versus the baseline entry; negative delta means smaller than baseline."""
    baseline = next(entry for entry in entries if entry["result_ref"] == baseline_ref)
    rows = []
    for entry in entries:
        values = {path: _normalized(entry, path, per_gpu) for path in metrics}
        values = {path: value for path, value in values.items() if value is not None}
        delta = {
            path: _delta(_normalized(baseline, path, per_gpu), value)
            for path, value in values.items()
            if entry is not baseline
        }
        rows.append(
            {
                "result_ref": entry["result_ref"],
                "label": entry.get("label"),
                "values": values,
                "delta": {path: value for path, value in delta.items() if value is not None},
            }
        )
    winners = {}
    for path in metrics:
        scored = [(row["values"][path], row["result_ref"]) for row in rows if path in row["values"]]
        if len(scored) > 1:
            lower = path.startswith(_LOWER_IS_BETTER)
            winners[path] = (min if lower else max)(scored)[1]
    return {"baseline_ref": baseline_ref, "rows": rows, "winner_by": winners}


def comparability(entries: list[dict[str, Any]], per_gpu: bool = False) -> dict[str, Any]:
    def distinct(key: str) -> list[Any]:
        return sorted({str(entry[key]) for entry in entries if entry.get(key) is not None})

    issues = []
    for key, kind, label in (
        ("cluster_id", "cluster_mismatch", "clusters"),
        ("model", "model_mismatch", "models"),
        ("hardware", "hardware_mismatch", "accelerator types"),
        ("workload_signature", "workload_mismatch", "workload parameters"),
        ("harness", "harness_version_mismatch", "benchmark harness versions"),
    ):
        values = distinct(key)
        if len(values) > 1:
            shown = values if key != "workload_signature" else [f"{len(values)} different workloads"]
            issues.append({"kind": kind, "message": f"The results use different {label}: {', '.join(shown)}."})
    if not per_gpu and len({entry.get("gpus") for entry in entries if entry.get("gpus")}) > 1:
        issues.append(
            {
                "kind": "gpu_budget_mismatch",
                "message": "The results used different accelerator counts; consider normalize='per_gpu'.",
            }
        )
    times = [entry["finished_epoch"] for entry in entries if entry.get("finished_epoch")]
    if times and max(times) - min(times) > _TIME_GAP_SECONDS:
        issues.append({"kind": "time_gap", "message": "The results were measured more than 7 days apart."})
    return {"comparable": not issues, "issues": issues}
