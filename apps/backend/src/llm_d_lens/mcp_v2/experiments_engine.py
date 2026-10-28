"""Experiment run engine.

An experiment is a persisted state machine stored in the ledger row's ``meta``. Lens runs the benchmarks and the
deployments server-side; ``advance`` only reads their state and issues the next step, so it can be called from any
process and any number of times. ``start_experiment``, ``get_experiment`` and ``cancel_experiment`` call it, and
never block longer than their own bounded wait. One benchmark runs at a time per experiment so variants do not
disturb each other's measurements.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import random
import time
import uuid
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .experiments_deploy import (
    LaunchUncertainError,
    adopt_created_run,
    delete_deployment,
    deployment_state,
    launch_spec,
)
from .experiments_results import (
    DEFAULT_COMPARE_METRICS,
    aggregate,
    comparability,
    compare_entries,
    sla_verdict,
    unit_result,
)
from .ledger import Resource
from .lens_api import LensApiError

DEPLOY_TIMEOUT_SECONDS = 1800
LAUNCH_BATCH = 3
CANCEL_WAIT_SECONDS = 20
CANCEL_TIMEOUT_SECONDS = 600
CREATE_RECONCILE_SECONDS = 600
MAX_CREATE_ATTEMPTS = 2
LEASE_TTL_SECONDS = 300
LEASE_WAIT_SECONDS = 10
MAX_EVENTS = 40
NOISY_SPREAD_PCT = 10.0
_TEXT_TAIL = 1500
_LOCKS: dict[str, asyncio.Lock] = {}
_DONE = ("succeeded", "failed", "cancelled", "skipped")


def _lock(exp_id: str) -> asyncio.Lock:
    return _LOCKS.setdefault(exp_id, asyncio.Lock())


class _LeaseLostError(Exception):
    """Another session took over the experiment after our lease expired; stop without writing."""


class _Lease:
    def __init__(self, ctx: ToolContext, exp_id: str):
        self.ctx, self.key, self.holder = ctx, f"exp:{exp_id}", uuid.uuid4().hex

    def renew(self) -> bool:
        return self.ctx.ledger.acquire_lease(self.key, self.holder, LEASE_TTL_SECONDS)

    def check(self) -> None:
        if not self.renew():
            raise _LeaseLostError


@contextlib.asynccontextmanager
async def _exclusive(ctx: ToolContext, exp_id: str, wait_seconds: float = 0.0):
    """In-process lock plus the ledger lease; yields the lease, or ``None`` if another session holds it."""
    async with _lock(exp_id):
        lease = _Lease(ctx, exp_id)
        deadline = time.monotonic() + wait_seconds
        while not lease.renew():
            if time.monotonic() >= deadline:
                yield None
                return
            await asyncio.sleep(1)
        try:
            yield lease
        finally:
            ctx.ledger.release_lease(lease.key, lease.holder)


def build_units(design: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [v["variant_id"] for v in design["variants"]]
    if design["order"] == "randomized":
        random.Random(design["order_seed"]).shuffle(variants)  # noqa: S311  (ordering, not security)
    workloads = [w["workload_id"] for w in design["workloads"]]
    reps = range(1, design["repetitions"] + 1)
    if design["order"] == "interleaved":
        keys = [(v, w, r) for r in reps for w in workloads for v in variants]
    else:
        keys = [(v, w, r) for v in variants for w in workloads for r in reps]
    return [
        {"id": f"{v}/{w}/{r}", "variant": v, "workload": w, "rep": r, "status": "pending", "run_id": None}
        for v, w, r in keys
    ]


def initial_state(
    exp_id: str, design: dict[str, Any], plan_id: str | None, unit_seconds: dict[str, Any]
) -> dict[str, Any]:
    now = time.time()
    return {
        "exp_id": exp_id,
        "design": design,
        "plan_id": plan_id,
        "phase": "running",
        "units": build_units(design),
        "variants": {
            v["variant_id"]: {
                "deploy": "borrowed" if v["kind"] == "existing" else "none",
                "dep_id": v.get("deployment_id"),
                "execution_id": None,
                "failure": None,
                "cleanup": "n/a" if v["kind"] == "existing" else "pending",
            }
            for v in design["variants"]
        },
        "cancel": None,
        "unit_seconds": unit_seconds,
        "started_at": now,
        "settled_at": None,
        "expires_at": now + design["ttl_hours"] * 3600,
        "last_error": None,
        "events": [],
    }


def _event(s: dict[str, Any], text: str) -> None:
    s["events"] = [*s["events"][-(MAX_EVENTS - 1) :], {"t": round(time.time()), "event": text}]


def _tail(text: Any) -> str | None:
    return text[-_TEXT_TAIL:] if isinstance(text, str) and text else None


def _running(s: dict[str, Any]) -> dict[str, Any] | None:
    return next((u for u in s["units"] if u["status"] == "running"), None)


def _skip_dead(s: dict[str, Any]) -> None:
    """Units of a variant whose deployment failed, or whose earlier run failed, can no longer run."""
    for vid, v in s["variants"].items():
        failed_unit = any(u["variant"] == vid and u["status"] == "failed" for u in s["units"])
        if v["deploy"] != "failed" and not failed_unit:
            continue
        reason = "deployment failed" if v["deploy"] == "failed" else "an earlier run of this variant failed"
        for unit in s["units"]:
            if unit["variant"] == vid and unit["status"] == "pending":
                unit.update(status="skipped", error={"message": f"Skipped: {reason}."})


async def _poll_deployments(ctx: ToolContext, s: dict[str, Any]) -> None:
    live = [(vid, v) for vid, v in s["variants"].items() if v["deploy"] == "deploying"]
    results = await asyncio.gather(
        *(deployment_state(ctx, ctx.ledger.get(v["dep_id"])) for _, v in live), return_exceptions=True
    )
    for (vid, v), result in zip(live, results, strict=True):
        if isinstance(result, LensApiError):
            continue
        if isinstance(result, BaseException):
            raise result
        if result["state"] == "ready":
            v.update(deploy="ready", execution_id=result["execution_id"])
            _event(s, f"{vid}: deployment ready")
        elif result["state"] in ("failed", "gone"):
            v.update(deploy="failed", failure=_tail(str(result.get("failure") or "The deployment disappeared.")))
            _event(s, f"{vid}: deployment failed")
        elif time.time() - v.get("deploy_started_at", time.time()) > DEPLOY_TIMEOUT_SECONDS:
            v.update(deploy="failed", failure="The deployment did not become ready in 30 minutes.")
            _event(s, f"{vid}: deployment timed out")


def _run_error(run: dict[str, Any]) -> dict[str, Any]:
    error = {"message": str(run.get("error") or "The benchmark run failed.")[:500]}
    text = {"stderr": _tail(run.get("stderr")), "stdout": _tail(run.get("stdout"))}
    return {**error, "untrusted_text": {key: value for key, value in text.items() if value}}


async def _poll_unit(ctx: ToolContext, s: dict[str, Any]) -> None:
    unit = _running(s)
    if unit is None:
        return
    try:
        run = await ctx.api.request("GET", f"/evaluate/runs/{unit['run_id']}")
    except LensApiError as error:
        if error.status != 404:
            raise
        unit.update(status="failed", error={"message": "Lens no longer has this benchmark run."})
        return
    status = run.get("status")
    if status == "succeeded":
        result = unit_result(run)
        rates = [(p.get("metrics") or {}).get("success_rate") for p in result["points"]] or [
            result["metrics"].get("success_rate")
        ]
        if rates and all(rate == 0 for rate in rates):
            message = (
                "Lens finished the benchmark but every request failed (success rate 0%), so there is no measurement. "
                "The load generator could not get answers from the deployment. Check get_logs and diagnose_deployment. "
                "A borrowed deployment served through a shared Gateway is not reachable by the benchmark; "
                "use an experiment-created spec variant instead."
            )
            unit.update(status="failed", error={"message": message}, finished_epoch=time.time())
            _event(s, f"{unit['id']}: all requests failed")
        else:
            unit.update(status="succeeded", result=result, finished_epoch=time.time())
            _event(s, f"{unit['id']}: succeeded")
    elif status == "failed":
        unit.update(status="failed", error=_run_error(run), finished_epoch=time.time())
        _event(s, f"{unit['id']}: failed")
    elif status == "cancelled":
        unit.update(status="cancelled", finished_epoch=time.time())
        _event(s, f"{unit['id']}: cancelled")


async def _start_unit(ctx: ToolContext, s: dict[str, Any]) -> None:
    if _running(s):
        return
    unit = next((u for u in s["units"] if u["status"] == "pending"), None)
    if unit is None:
        return
    variant = s["variants"][unit["variant"]]
    if variant["deploy"] == "borrowed":
        state = await deployment_state(ctx, ctx.ledger.get(variant["dep_id"]))
        if state["state"] != "ready":
            variant.update(deploy="failed", failure=f"The existing deployment is {state['state']}, not ready.")
            return
        execution_id = state["execution_id"]
    elif variant["deploy"] == "ready":
        execution_id = variant["execution_id"]
    else:
        return
    workload = next(w for w in s["design"]["workloads"] if w["workload_id"] == unit["workload"])
    body = {"deployment_execution_id": execution_id, **workload["benchmark"]}
    if workload["sla"]:
        body["sla_targets"] = workload["sla"]
    try:
        run = await ctx.api.request("POST", "/evaluate/runs", body=body, timeout=60)
    except LensApiError as error:
        if error.status >= 500 or error.status == 429:
            raise
        unit.update(status="failed", error={"message": error.message()[:500]}, finished_epoch=time.time())
        return
    unit.update(status="running", run_id=run["id"], started_at=time.time())
    _event(s, f"{unit['id']}: benchmark started")


async def _launch_one(ctx: ToolContext, s: dict[str, Any], vid: str) -> None:
    design, variant = s["design"], s["variants"][vid]
    spec = next(v["spec"] for v in design["variants"] if v["variant_id"] == vid)
    variant["create_attempts"] = variant.get("create_attempts", 0) + 1
    try:
        row = await launch_spec(
            ctx,
            cluster_id=design["cluster_id"],
            spec=spec,
            root_id=s["exp_id"],
            variant_id=vid,
            expires_at=s["expires_at"],
        )
    except LaunchUncertainError as error:
        variant.update(deploy="creating", dep_id=error.row_id, deploy_started_at=time.time())
        _event(s, f"{vid}: deployment creation unconfirmed, reconciling")
        return
    except ToolError as error:
        variant.update(deploy="failed", failure=error.message[:500])
        _event(s, f"{vid}: deployment could not be created")
        return
    except LensApiError as error:
        variant.update(deploy="failed", failure=error.message()[:500])
        _event(s, f"{vid}: deployment could not be created")
        return
    variant.update(deploy="deploying", dep_id=row.id, deploy_started_at=time.time())
    _event(s, f"{vid}: deployment {row.id} started")


async def _reconcile(ctx: ToolContext, s: dict[str, Any]) -> None:
    """Resolve deployments whose creation was unconfirmed (timeout, 5xx, cancelled call) before creating again."""
    children = ctx.ledger.children(s["exp_id"])
    for vid, v in s["variants"].items():
        orphan = next((c for c in children if c.meta.get("variant_id") == vid), None)
        if v["deploy"] == "none" and orphan is not None:
            started = orphan.meta.get("launch_epoch", time.time())
            v.update(
                deploy="deploying" if orphan.lens_run_id else "creating", dep_id=orphan.id, deploy_started_at=started
            )
    for vid, v in s["variants"].items():
        if v["deploy"] != "creating":
            continue
        row = ctx.ledger.get(v["dep_id"])
        if row is None or row.state == "deleted":
            v.update(deploy="none", dep_id=None)
            continue
        try:
            found = bool(row.lens_run_id) or await adopt_created_run(ctx, row)
        except LensApiError:
            continue
        if found:
            v.update(deploy="deploying", deploy_started_at=time.time())
            _event(s, f"{vid}: deployment {row.id} confirmed")
        elif time.time() - row.meta.get("launch_epoch", v.get("deploy_started_at", 0)) > CREATE_RECONCILE_SECONDS:
            ctx.ledger.set_state(row.id, "deleted")
            if v.get("create_attempts", 1) >= MAX_CREATE_ATTEMPTS:
                v.update(deploy="failed", dep_id=None, failure="Lens never confirmed the deployment creation.")
            else:
                v.update(deploy="none", dep_id=None)
            _event(s, f"{vid}: unconfirmed deployment {row.id} abandoned")


def _live_spec(s: dict[str, Any]) -> bool:
    return any(
        v["dep_id"] and v["cleanup"] in ("pending", "failed") and v["deploy"] != "borrowed"
        for v in s["variants"].values()
    )


async def _launch(ctx: ToolContext, s: dict[str, Any]) -> None:
    order = list(dict.fromkeys(u["variant"] for u in s["units"] if u["status"] == "pending"))
    waiting = [vid for vid in order if s["variants"][vid]["deploy"] == "none"]
    if s["design"]["schedule"] == "sequential":
        if _live_spec(s):
            return
        waiting = waiting[:1]
    else:
        waiting = waiting[:LAUNCH_BATCH]
    await asyncio.gather(*(_launch_one(ctx, s, vid) for vid in waiting))


async def _cleanup(ctx: ToolContext, s: dict[str, Any]) -> None:
    if (s["cancel"] or {}).get("defer_cleanup"):
        return
    todo = []
    for vid, v in s["variants"].items():
        busy = any(u["variant"] == vid and u["status"] in ("pending", "running") for u in s["units"])
        if v["cleanup"] == "pending" and not v["dep_id"] and v["deploy"] in ("none", "failed") and not busy:
            v["cleanup"] = "n/a"
        if (
            v["cleanup"] in ("pending", "failed")
            and v["dep_id"]
            and v["deploy"] not in ("none", "creating")
            and not busy
        ):
            todo.append((vid, v))

    async def one(vid: str, v: dict[str, Any]) -> None:
        row = ctx.ledger.get(v["dep_id"])
        if s["design"]["keep_deployments"] or v.get("keep"):
            v["cleanup"] = "kept"
            ctx.ledger.bind_root(v["dep_id"], v["dep_id"])
            return
        if row is None or row.state == "deleted":
            v["cleanup"] = "done"
            return
        try:
            deleted = await delete_deployment(ctx, row)
        except LensApiError as error:
            v.update(cleanup="failed", residual=error.message()[:300])
            return
        if deleted:
            v.update(cleanup="done", deploy="deleted", residual=None)
            _event(s, f"{vid}: temporary deployment deleted")
        else:
            v.update(cleanup="failed", residual="Lens still reports the deployment in use by a benchmark.")

    await asyncio.gather(*(one(vid, v) for vid, v in todo))


async def _apply_cancel(ctx: ToolContext, s: dict[str, Any]) -> None:
    cancel = s["cancel"]
    targets = set(cancel["variant_ids"] or s["variants"])
    for unit in s["units"]:
        if unit["variant"] in targets and unit["status"] == "pending":
            unit["status"] = "cancelled"
    unit = _running(s)
    target_running = unit is not None and unit["variant"] in targets
    if target_running and time.time() - cancel["at"] > CANCEL_TIMEOUT_SECONDS:
        unit.update(
            status="cancelled",
            local_cancel=True,
            finished_epoch=time.time(),
            error={"message": "Lens did not confirm the cancellation; marked cancelled locally."},
        )
        warning = (
            f"Benchmark run {unit['run_id']} did not reach a terminal state "
            f"{CANCEL_TIMEOUT_SECONDS // 60} minutes after cancelling; it is marked cancelled locally and may still "
            "hold its deployment (cleanup then reports a residual)."
        )
        s["warnings"] = [*s.get("warnings", []), warning]
        _event(s, f"{unit['id']}: cancellation unconfirmed, marked cancelled locally")
    elif target_running:
        if not unit.get("cancel_sent"):
            try:
                await ctx.api.request("POST", f"/evaluate/runs/{unit['run_id']}/cancel", timeout=60)
            except LensApiError as error:
                if error.status not in (404, 409):
                    raise
            unit["cancel_sent"] = True
        await _poll_unit(ctx, s)
    if cancel["variant_ids"] and not (target_running and unit["status"] == "running"):
        s["cancel"] = None
        s["phase"] = "running"


def _settle(s: dict[str, Any]) -> None:
    if s["phase"] != "settled" and all(u["status"] in _DONE for u in s["units"]):
        s.update(phase="settled", settled_at=time.time())
        _event(s, "experiment settled")


def has_residuals(s: dict[str, Any]) -> bool:
    """Cleanup that failed, or (once settled) a deployment whose teardown has not happened yet."""
    return any(
        v["cleanup"] == "failed" or (s["phase"] == "settled" and v["cleanup"] == "pending" and v["dep_id"])
        for v in s["variants"].values()
    )


async def _step(ctx: ToolContext, s: dict[str, Any], allow_launch: bool, lease: _Lease) -> None:
    lease.check()
    await _reconcile(ctx, s)
    if s["phase"] == "settled":
        await _cleanup(ctx, s)
        return
    lease.check()
    if s["cancel"]:
        await _apply_cancel(ctx, s)
    else:
        await _poll_deployments(ctx, s)
        await _poll_unit(ctx, s)
        _skip_dead(s)
        lease.check()
        await _start_unit(ctx, s)
        lease.check()
        if allow_launch:
            await _launch(ctx, s)
    _skip_dead(s)
    lease.check()
    await _cleanup(ctx, s)
    _settle(s)


async def _advance_locked(ctx: ToolContext, exp_id: str, allow_launch: bool, lease: _Lease) -> Resource | None:
    """Read-modify-write of the experiment state; the caller holds the lease for the whole call."""
    row = ctx.ledger.get(exp_id)
    if row is None or row.state == "deleted" or "phase" not in row.meta:
        return row
    if row.meta["phase"] == "settled" and not has_residuals(row.meta):
        return row
    s = copy.deepcopy(row.meta)
    lost = False
    try:
        await _step(ctx, s, allow_launch, lease)
        s["last_error"] = None
    except (LensApiError, ToolError) as error:
        s["last_error"] = (
            (getattr(error, "message", None) or str(error)) if isinstance(error, ToolError) else error.message()
        )
    except _LeaseLostError:
        lost = True
    finally:
        if not lost and lease.renew():
            ctx.ledger.update_meta(exp_id, s)
            if s["phase"] == "settled":
                ctx.ledger.set_state(exp_id, "completed")
    return ctx.ledger.get(exp_id)


async def advance(ctx: ToolContext, exp_id: str, *, allow_launch: bool = True) -> tuple[Resource | None, bool]:
    """One bounded step. Returns ``(row, contended)``; contended means another session holds the experiment's
    lease, so nothing was advanced and ``row`` is the last persisted state."""
    async with _exclusive(ctx, exp_id) as lease:
        if lease is None:
            return ctx.ledger.get(exp_id), True
        return await _advance_locked(ctx, exp_id, allow_launch, lease), False


def fingerprint(s: dict[str, Any]) -> str:
    return json.dumps(
        [s["phase"], [u["status"] for u in s["units"]], [[v["deploy"], v["cleanup"]] for v in s["variants"].values()]]
    )


async def wait_for_change(
    ctx: ToolContext, exp_id: str, wait_seconds: float, until: str
) -> tuple[Resource | None, bool]:
    deadline = time.monotonic() + wait_seconds
    row, contended = await advance(ctx, exp_id)
    first = fingerprint(row.meta) if row else ""
    while row and wait_seconds > 0:
        if row.meta.get("phase") == "settled" and not has_residuals(row.meta):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(3.0, remaining))
        row, contended = await advance(ctx, exp_id, allow_launch=deadline - time.monotonic() > 20)
        if row and until == "change" and fingerprint(row.meta) != first:
            break
    return row, contended


async def request_cancel(
    ctx: ToolContext,
    exp_id: str,
    *,
    variant_ids: list[str] | None,
    keep_deployments: bool,
    reason: str | None,
    wait_seconds: float | None = None,
    defer_cleanup: bool = False,
) -> Resource | None:
    """Cancel -> wait for the Lens run to be terminal -> delete temporary deployments, in that order."""
    async with _exclusive(ctx, exp_id, LEASE_WAIT_SECONDS) as lease:
        if lease is None:
            raise ToolError(
                "conflict_state",
                f"Another session is currently advancing {exp_id}.",
                hint="Retry cancel_experiment in a few seconds.",
            )
        row = ctx.ledger.get(exp_id)
        if row is None or row.meta.get("phase") == "settled":
            return await _advance_locked(ctx, exp_id, False, lease)
        previous = row.meta.get("cancel") or {}
        cancel = {
            "variant_ids": variant_ids,
            "keep_deployments": keep_deployments,
            "defer_cleanup": defer_cleanup,
            "reason": reason,
            "at": previous.get("at") or time.time(),
        }
        variants = copy.deepcopy(row.meta["variants"])
        if keep_deployments:
            for vid in variant_ids or variants:
                variants[vid]["keep"] = True
        patch: dict[str, Any] = {"cancel": cancel, "variants": variants}
        if not variant_ids:
            patch["phase"] = "cancelling"
        ctx.ledger.update_meta(exp_id, patch)
        deadline = time.monotonic() + (CANCEL_WAIT_SECONDS if wait_seconds is None else wait_seconds)
        while True:
            row = await _advance_locked(ctx, exp_id, False, lease)
            settled = row.meta["phase"] == "settled" and not has_residuals(row.meta)
            if settled or row.meta["cancel"] is None or time.monotonic() >= deadline:
                return row
            await asyncio.sleep(2)


def unit_counts(s: dict[str, Any]) -> dict[str, int]:
    counts = dict.fromkeys(("pending", "running", "succeeded", "failed", "cancelled", "skipped"), 0)
    for unit in s["units"]:
        counts[unit["status"]] += 1
    return counts


def signature(benchmark: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(benchmark, sort_keys=True).encode()).hexdigest()[:12]


def result_entries(row: Resource) -> list[dict[str, Any]]:
    """One aggregated result per (variant, workload) that has at least one successful repetition."""
    s, design, entries = row.meta, row.meta["design"], []
    for variant in design["variants"]:
        for workload in design["workloads"]:
            done = [
                u
                for u in s["units"]
                if u["variant"] == variant["variant_id"]
                and u["workload"] == workload["workload_id"]
                and u["status"] == "succeeded"
            ]
            if not done:
                continue
            aggregated = aggregate([u["result"] for u in done])
            entries.append(
                {
                    **aggregated,
                    "result_ref": f"{row.id}#{variant['variant_id']}/{workload['workload_id']}",
                    "experiment_id": row.id,
                    "variant_id": variant["variant_id"],
                    "workload_id": workload["workload_id"],
                    "label": variant.get("label") or variant["variant_id"],
                    "gpus": variant["gpus"] or None,
                    "workload_signature": signature(workload["benchmark"]),
                    "cluster_id": aggregated.get("cluster_id") or design["cluster_id"],
                    "model": aggregated.get("model") or (variant.get("spec") or {}).get("model"),
                    "finished_epoch": max(u.get("finished_epoch") or 0 for u in done),
                    "sla": sla_verdict(aggregated["metrics"], workload["sla"]),
                    "planned_repetitions": design["repetitions"],
                }
            )
    return entries


def execution_state(s: dict[str, Any]) -> str:
    counts = unit_counts(s)
    if s["phase"] == "cancelling":
        return "cancelling"
    if s["phase"] != "settled":
        return (
            "queued" if not s["units"] or not any(counts[k] for k in ("running", "succeeded", "failed")) else "running"
        )
    if s["cancel"] and not s["cancel"]["variant_ids"]:
        return "cancelled"
    if counts["succeeded"] == len(s["units"]):
        return "succeeded"
    return "partially_succeeded" if counts["succeeded"] else "failed"


def cleanup_state(s: dict[str, Any]) -> dict[str, Any]:
    states = [v["cleanup"] for v in s["variants"].values() if v["cleanup"] != "n/a"]
    residuals = [
        {"deployment_id": v["dep_id"], "variant_id": vid, "reason": v.get("residual")}
        for vid, v in s["variants"].items()
        if v["cleanup"] == "failed"
    ]
    if not states:
        state = "not_needed"
    elif residuals:
        state = "failed"
    elif all(item in ("done", "kept") for item in states):
        state = "kept" if "kept" in states else "done"
    else:
        state = "pending"
    result: dict[str, Any] = {"state": state}
    if residuals:
        result["residuals"] = residuals
        result["hint"] = "Retry by calling get_experiment, or delete_resources on the listed deployment ids."
    return result


def _measurement(s: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = unit_counts(s)
    total = len(s["units"])
    if counts["succeeded"] == total:
        state = "complete"
    elif counts["succeeded"]:
        state = "partial"
    else:
        state = "pending" if s["phase"] != "settled" else "none"
    measurement: dict[str, Any] = {"state": state, "valid_runs": counts["succeeded"], "planned_runs": total}
    noisy = [e["result_ref"] for e in entries if (e.get("throughput_spread_pct") or 0) > NOISY_SPREAD_PCT]
    if noisy:
        measurement["noisy_results"] = noisy
    return measurement


def _mechanism(s: dict[str, Any]) -> dict[str, Any]:
    collected = any((u.get("result") or {}).get("mechanism_collected") for u in s["units"])
    if collected:
        return {"state": "collected", "note": "Server-side counters are stored with the Lens benchmark runs."}
    return {"state": "pending" if s["phase"] != "settled" else "not_collected"}


def _variant_view(row: Resource, s: dict[str, Any], vid: str, is_mine: bool) -> dict[str, Any]:
    v, spec = s["variants"][vid], next(x for x in s["design"]["variants"] if x["variant_id"] == vid)
    counts = dict.fromkeys(("pending", "running", "succeeded", "failed", "cancelled", "skipped"), 0)
    for unit in s["units"]:
        if unit["variant"] == vid:
            counts[unit["status"]] += 1
    view: dict[str, Any] = {
        "variant_id": vid,
        "label": spec.get("label"),
        "kind": "existing_deployment" if spec["kind"] == "existing" else "temporary_deployment",
        "deployment_id": v["dep_id"],
        "deployment_state": v["deploy"],
        "runs": {key: value for key, value in counts.items() if value},
    }
    if spec["kind"] == "existing":
        view["note"] = "borrowed: never modified or deleted by the experiment"
    if v.get("failure"):
        view["untrusted_text"] = {"failure": v["failure"]}
    errors = [u["error"] for u in s["units"] if u["variant"] == vid and u.get("error") and u["status"] == "failed"]
    if errors:
        view["error"] = errors[0]
    return view


def _eta(s: dict[str, Any]) -> int | None:
    if s["phase"] == "settled":
        return 0
    seconds = 0
    for unit in s["units"]:
        if unit["status"] in ("pending", "running"):
            estimate = s["unit_seconds"].get(unit["workload"])
            if estimate is None:
                return None
            seconds += estimate
    pending_deploys = sum(1 for v in s["variants"].values() if v["deploy"] in ("none", "deploying"))
    return seconds + pending_deploys * 600


def build_view(row: Resource, include: list[str], is_mine: bool = True, contended: bool = False) -> dict[str, Any]:
    s = row.meta
    counts = unit_counts(s)
    total = len(s["units"])
    finished = total - counts["pending"] - counts["running"]
    entries = result_entries(row)
    view: dict[str, Any] = {
        "experiment_id": row.id,
        "name": row.name,
        "cluster_id": row.cluster_id,
        "is_mine": is_mine,
        "phase": s["phase"],
        "state": {
            "execution": execution_state(s),
            "measurement": _measurement(s, entries),
            "mechanism": _mechanism(s),
            "cleanup": cleanup_state(s),
        },
        "progress": {
            "runs": counts,
            "percent": round(100 * finished / total) if total else 100,
            "eta_seconds": _eta(s),
            "benchmark_runs_total": total,
        },
        "variants": [_variant_view(row, s, v["variant_id"], is_mine) for v in s["design"]["variants"]],
        "expires_at": s["expires_at"],
    }
    if s.get("last_error"):
        view["advance_error"] = s["last_error"]
    if s.get("warnings"):
        view["warnings"] = s["warnings"]
    if s["phase"] != "settled":
        view["next"] = "Poll get_experiment (wait_seconds up to 25); progress advances while you poll."
    elif execution_state(s) in ("succeeded", "partially_succeeded"):
        view["next"] = "Read results with get_experiment(include=['results','comparison']) or compare_results."
    if contended:
        view["advancing_elsewhere"] = True
        view["note"] = "Another session is advancing this experiment; this is the last persisted state. Retry shortly."
    if "results" in include or "points" in include:
        view["results"] = [_result_view(entry, "points" in include) for entry in entries]
    if "comparison" in include:
        view["comparison"] = _comparison(entries, s["design"])
    if "timeline" in include:
        view["timeline"] = s["events"]
    return view


def _result_view(entry: dict[str, Any], with_points: bool) -> dict[str, Any]:
    view = {
        key: entry[key]
        for key in (
            "result_ref",
            "variant_id",
            "workload_id",
            "label",
            "metrics",
            "headline_point",
            "repetitions",
            "planned_repetitions",
            "throughput_spread_pct",
            "sla",
        )
        if entry.get(key) is not None
    }
    if with_points:
        view["points"] = entry["points"]
    return view


def _comparison(entries: list[dict[str, Any]], design: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = []
    for workload in design["workloads"]:
        group = [e for e in entries if e["workload_id"] == workload["workload_id"]]
        if len(group) < 2:
            continue
        comparisons.append(
            {
                "workload_id": workload["workload_id"],
                **compare_entries(group, group[0]["result_ref"], DEFAULT_COMPARE_METRICS),
                "comparability": comparability(group),
            }
        )
    return comparisons


async def describe_experiment(ctx: ToolContext, row: Resource) -> dict[str, Any]:
    s = row.meta
    if "phase" not in s:
        return {"status": "unknown"}
    counts = unit_counts(s)
    return {
        "status": execution_state(s),
        "progress": f"{len(s['units']) - counts['pending'] - counts['running']}/{len(s['units'])}",
        "variants": len(s["variants"]),
    }


async def dependents_of_experiment(ctx: ToolContext, row: Resource) -> list[dict[str, Any]]:
    return []


async def delete_experiment(ctx: ToolContext, row: Resource, cascade: bool) -> None:
    """Cancel if running, wait for a terminal state, delete Lens run records, then the temporary deployments."""
    if row.meta.get("phase") in ("running", "cancelling"):
        row = (
            await request_cancel(
                ctx, row.id, variant_ids=None, keep_deployments=False, reason="deleted", defer_cleanup=True
            )
            or row
        )
        if row.meta.get("phase") != "settled":
            raise ToolError(
                "in_use",
                f"{row.id} is still stopping its benchmark run.",
                hint="Retry delete_resources in a minute, or check get_experiment.",
            )
    for unit in row.meta.get("units", []):
        if not unit.get("run_id"):
            continue
        local_cancel = bool(unit.get("local_cancel"))
        try:
            await ctx.api.request("DELETE", f"/evaluate/runs/{unit['run_id']}", timeout=60)
        except LensApiError as error:
            if error.status == 409 and not local_cancel:
                raise ToolError(
                    "in_use", f"Benchmark run {unit['run_id']} is not finished yet.", hint="Retry shortly."
                ) from error
            if error.status not in (404, 409):
                raise
    for child in ctx.ledger.children(row.id):
        if child.kind != "deployment":
            continue
        if child.state == "creating" and not child.lens_run_id:
            with contextlib.suppress(LensApiError):
                await adopt_created_run(ctx, child)
            child = ctx.ledger.get(child.id) or child
            age = time.time() - child.meta.get("launch_epoch", 0)
            if not child.lens_run_id and age < CREATE_RECONCILE_SECONDS:
                raise ToolError(
                    "in_use",
                    f"Creation of temporary deployment {child.id} is unconfirmed.",
                    hint="Retry delete_resources in a few minutes.",
                )
        if not await delete_deployment(ctx, child):
            raise ToolError(
                "in_use",
                f"Temporary deployment {child.id} is still in use by Lens.",
                hint="Retry delete_resources in a minute.",
            )
