"""Rows of measurements from Lens evaluation workflow details, and their comparison."""

from __future__ import annotations

import re
import statistics
from typing import Any

from .experiments_results import DEFAULT_COMPARE_METRICS, _median_nested, get_metric, sla_verdict, unit_result

_DESCRIPTION = re.compile(r"^lens-mcp variant=(?P<variant>\S+) rep=(?P<k>\d+)/(?P<n>\d+)")
_SCENARIO_NAME = re.compile(r"^.* \(r(?P<k>\d+)/\d+\)$")
LOWER_IS_BETTER = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_latency_ms")


def _plan_labels(workflow: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """plan id -> (variant id, repetition) for plans this server created; other plans keep their own id."""
    labels = {}
    for plan in workflow.get("benchmark_plans") or []:
        match = _DESCRIPTION.match(plan.get("deployment_description") or "")
        labels[plan["id"]] = (match["variant"], int(match["k"])) if match else (plan["id"], 1)
    return labels


def _scenario_base(case: dict[str, Any]) -> tuple[str, int]:
    scenario_id, name = case.get("scenario_id") or "", case.get("scenario_name") or ""
    match = _SCENARIO_NAME.match(name)
    if match and scenario_id.endswith(f"-r{match['k']}"):
        return scenario_id[: -len(f"-r{match['k']}")], int(match["k"])
    return scenario_id, 1


def case_rows(details: dict[str, Any], source: str) -> list[dict[str, Any]]:
    labels = _plan_labels(details.get("workflow") or {})
    rows = []
    for entry in details.get("cases") or []:
        case, run = entry.get("case") or {}, entry.get("evaluation") or {}
        variant, plan_repetition = labels.get(case.get("benchmark_plan_id"), (case.get("benchmark_plan_id"), 1))
        scenario, scenario_repetition = _scenario_base(case)
        baseline_type = case.get("baseline_type") if case.get("kind") == "baseline" else None
        row: dict[str, Any] = {
            "source": source,
            "case_id": case.get("id"),
            "kind": case.get("kind"),
            "baseline_type": baseline_type,
            "plan": variant if baseline_type is None else f"{variant}+{baseline_type}",
            "scenario": scenario or None,
            "repetition": plan_repetition if plan_repetition > 1 else scenario_repetition,
            "status": case.get("status"),
            "run_id": case.get("evaluation_run_id"),
            "sla_targets": case.get("sla_targets") or {},
            "error": case.get("error"),
        }
        if run.get("status") == "succeeded":
            unit = unit_result(run)
            row.update(metrics=unit["metrics"], headline_point=unit["headline_point"], points=unit["points"])
        rows.append(row)
    return rows


def run_row(run: dict[str, Any]) -> dict[str, Any]:
    """A standalone benchmark run as a single-repetition row."""
    unit = unit_result(run)
    return {
        "source": run.get("id"),
        "case_id": None,
        "kind": "benchmark_run",
        "baseline_type": None,
        "plan": run.get("id"),
        "scenario": None,
        "repetition": 1,
        "status": run.get("status"),
        "run_id": run.get("id"),
        "sla_targets": {},
        "error": run.get("error"),
        "metrics": unit["metrics"],
        "headline_point": unit["headline_point"],
        "points": unit["points"],
    }


def result_view(rows: list[dict[str, Any]], include_points: bool) -> list[dict[str, Any]]:
    view = []
    for row in rows:
        entry = {
            key: row[key]
            for key in ("case_id", "plan", "kind", "baseline_type", "scenario", "repetition", "status", "run_id")
            if row.get(key) is not None
        }
        if row.get("metrics"):
            entry["metrics"] = row["metrics"]
            entry["sla"] = sla_verdict(row["metrics"], row.get("sla_targets"))
            if row.get("headline_point"):
                entry["headline_point"] = row["headline_point"]
            if include_points:
                entry["points"] = row["points"]
        elif row.get("error"):
            entry["error"] = str(row["error"])[:300]
        view.append(entry)
    return view


def _point_key(point: dict[str, Any]) -> tuple:
    return tuple((key, point.get(key)) for key in ("isl", "osl", "concurrency", "rate_rps"))


def _label(key: tuple) -> dict[str, Any]:
    identity = dict(key)
    result: dict[str, Any] = {}
    concurrency_or_rate = (
        f"c{identity['concurrency']}"
        if identity.get("concurrency") is not None
        else f"rps{identity['rate_rps']}"
        if identity.get("rate_rps") is not None
        else None
    )
    if concurrency_or_rate:
        result["concurrency_or_rate"] = concurrency_or_rate
    if identity.get("isl") is not None:
        result["workload"] = f"{identity['isl']}/{identity.get('osl')}"
    return result


def compare(
    rows: list[dict[str, Any]], baseline: str | None, metrics: list[str] | None, group_repetitions: bool
) -> dict[str, Any]:
    paths = metrics or DEFAULT_COMPARE_METRICS
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        for point in row.get("points") or [{"metrics": row.get("metrics") or {}}]:
            if not point.get("metrics"):
                continue
            repetition = None if group_repetitions else row["repetition"]
            key = (row["source"], row["plan"], row["scenario"], repetition, _point_key(point))
            groups.setdefault(key, []).append({"metrics": point["metrics"], "sla": row.get("sla_targets")})
    labels = list(dict.fromkeys(key[1] for key in groups))
    baseline = baseline or (labels[0] if labels else None)
    if baseline is not None and baseline not in labels:
        return {"error": "baseline_not_found", "plans": labels}
    medians = {key: _median_nested([item["metrics"] for item in items]) for key, items in groups.items()}
    baseline_source = next((key[0] for key in groups if key[1] == baseline), None)
    reference = {
        (key[2], key[3], key[4]): medians[key] for key in groups if key[1] == baseline and key[0] == baseline_source
    }
    table = []
    for key, items in groups.items():
        source, plan, scenario, repetition, point = key
        verdict = sla_verdict(medians[key], items[0]["sla"])["verdict"]
        for path in paths:
            values = [value for item in items if (value := get_metric(item["metrics"], path)) is not None]
            if not values:
                continue
            median = statistics.median(values)
            base = get_metric(reference.get((scenario, repetition, point), {}), path)
            row = {
                "source": source,
                "plan": plan,
                "scenario": scenario,
                **_label(point),
                "metric": path,
                "n": len(values),
                "median": round(median, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "delta_vs_baseline_pct": None
                if base in (None, 0) or plan == baseline
                else round((median - base) / base * 100, 1),
                "sla_pass": {"pass": True, "fail": False}.get(verdict),
            }
            if repetition is not None:
                row["repetition"] = repetition
            table.append(row)
    return {"baseline": baseline, "rows": table, "lower_is_better": [m for m in paths if m.startswith(LOWER_IS_BETTER)]}
