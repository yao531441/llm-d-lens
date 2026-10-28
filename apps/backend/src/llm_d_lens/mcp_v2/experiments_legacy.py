"""Frozen read/cancel/compare path for legacy ``exp-`` experiments; the engine modules it calls are unchanged."""

from __future__ import annotations

import re
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .experiments_engine import build_view, request_cancel, result_entries, wait_for_change
from .experiments_results import DEFAULT_COMPARE_METRICS, comparability, compare_entries
from .ledger import Resource

LEGACY_PREFIX = "exp-"
RESULT_REF = re.compile(r"^exp-[a-z0-9]+#[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
LEGACY_MAX_WAIT = 25


def is_legacy(experiment_id: str) -> bool:
    return experiment_id.startswith(LEGACY_PREFIX)


def find_row(ctx: ToolContext, experiment_id: str) -> Resource:
    row = ctx.ledger.get(experiment_id)
    if row is None or row.kind != "experiment" or row.state == "deleted":
        raise ToolError(
            "not_found",
            f"No experiment '{experiment_id}'.",
            field="experiment_id",
            hint="Call list_resources(kinds=['experiment']) for ids.",
        )
    return row


async def get(ctx: ToolContext, experiment_id: str, include: list[str], wait_seconds: int, wait_for: str):
    row = find_row(ctx, experiment_id)
    if "phase" not in row.meta:
        raise ToolError("conflict_state", f"{row.id} is still being created.", hint="Retry in a few seconds.")
    mine = row.owner == ctx.owner
    contended = False
    if mine:
        advanced, contended = await wait_for_change(
            ctx,
            row.id,
            min(wait_seconds, LEGACY_MAX_WAIT),
            "settled" if wait_for == "terminal" else "change",
        )
        row = advanced or row
    view = build_view(row, [item for item in include if item != "cases"], is_mine=mine, contended=contended)
    view["legacy"] = True
    return view


async def cancel(ctx: ToolContext, experiment_id: str, reason: str | None) -> dict[str, Any]:
    row = find_row(ctx, experiment_id)
    if row.owner != ctx.owner:
        raise ToolError(
            "not_owner",
            f"{row.id} belongs to another user.",
            field="experiment_id",
            hint="You can read it with get_experiment but not cancel it.",
        )
    row = await request_cancel(ctx, row.id, variant_ids=None, keep_deployments=False, reason=reason) or row
    view = build_view(row, ["results"])
    view["legacy"] = True
    view["cancelled"] = row.meta["phase"] == "settled"
    if not view["cancelled"]:
        view["next"] = "Teardown is still running; call get_experiment to follow it (it finishes while you poll)."
    return view


def entry_for(ctx: ToolContext, ref: str) -> dict[str, Any]:
    if not RESULT_REF.match(ref):
        raise ToolError(
            "invalid_argument",
            f"'{ref}' is not a result reference.",
            field="result_refs",
            hint="Format: exp-xxxxxxxx#variant_id/workload_id.",
        )
    row = ctx.ledger.get(ref.split("#", 1)[0])
    if row is None or row.kind != "experiment" or row.state == "deleted" or "design" not in row.meta:
        raise ToolError(
            "not_found",
            f"No experiment for result reference '{ref}'.",
            field="result_refs",
            hint="Use references from get_experiment(include=['results']).",
        )
    entry = next((item for item in result_entries(row) if item["result_ref"] == ref), None)
    if entry is None:
        raise ToolError(
            "conflict_state",
            f"'{ref}' has no finished measurement (yet, or its runs failed).",
            field="result_refs",
            hint="Check get_experiment(include=['results']) for the references that exist.",
        )
    return entry


def compare(ctx: ToolContext, refs: list[str], baseline_ref: str | None, metrics: list[str] | None, normalize: str):
    refs = list(dict.fromkeys(refs))
    if len(refs) < 2:
        raise ToolError("invalid_argument", "Pass at least two different result references.", field="result_refs")
    baseline = baseline_ref or refs[0]
    if baseline not in refs:
        raise ToolError(
            "invalid_argument",
            "baseline_ref must be one of result_refs.",
            field="baseline_ref",
            hint="Add the baseline to result_refs or pick one of them.",
        )
    entries = [entry_for(ctx, ref) for ref in refs]
    per_gpu = normalize == "per_gpu"
    return {
        "legacy": True,
        "comparability": comparability(entries, per_gpu),
        "comparison": compare_entries(entries, baseline, list(metrics or DEFAULT_COMPARE_METRICS), per_gpu),
        "results": [
            {key: entry[key] for key in ("result_ref", "label", "metrics", "repetitions", "gpus", "sla")}
            for entry in entries
        ],
    }
