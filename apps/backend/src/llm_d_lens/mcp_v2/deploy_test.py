"""test_deployment engine: sends a few real requests to the caller's deployment and checks what Lens can observe.

Lens has no deployment verification endpoint, so the checks combine the model response itself with request counts
read from the model servers' access logs before and after. Checks that cannot be proven from that report
``inconclusive`` with ``evidence_sufficient=false`` instead of guessing.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from .context import ToolContext
from .deploy_spec import PD_GUIDE
from .deploy_status import default_container, now_iso
from .deployment_refs import DeploymentRef
from .lens_api import LensApiError

CHECKS = ("chat", "pd_transfer", "prefix_cache", "routing_spread", "rank_distribution")
PREFIX_REQUESTS = 6
LOG_WINDOW = 2000
REQUEST_TIMEOUT = 45.0
CONCURRENCY = 4
ACCESS_LINE = re.compile(r"POST /v1/chat/completions")
SPREAD_FAIL_MIN_REQUESTS = 10
SHARED_PREFIX = " ".join(f"Rule {i}: answer briefly and stay on topic." for i in range(60))
TOPICS = ("rivers", "bridges", "bread", "clouds", "violins", "tides", "maps", "lanterns", "glaciers", "orchards")


class EndpointUnreachableError(Exception):
    pass


async def http_call(method: str, url: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, Any] | None]:
    """One HTTP request to the forwarded deployment endpoint; tests replace this function."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, trust_env=False) as client:
            response = await client.request(method, url, json=payload)
    except httpx.HTTPError as error:
        raise EndpointUnreachableError(type(error).__name__) from error
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _chat_payload(model: str, prompt: str) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0}


async def _send_all(url: str, payloads: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any] | None]]:
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        async with gate:
            return await http_call("POST", f"{url}/v1/chat/completions", payload)

    return list(await asyncio.gather(*(one(payload) for payload in payloads)))


async def _request_counts(
    ctx: ToolContext, execution_id: str, pods: list[dict[str, Any]]
) -> dict[str, tuple[int, bool]]:
    """Per pod: access-log lines for chat completions in the log window, and whether the window may have rolled."""
    counts: dict[str, tuple[int, bool]] = {}
    for pod in pods:
        try:
            data = await ctx.api.request(
                "GET",
                f"/deployments/executions/{execution_id}/pods/{pod['name']}/logs",
                params={"container": default_container(pod), "tail": LOG_WINDOW},
                timeout=40,
            )
        except LensApiError:
            continue
        if data.get("success"):
            lines = (data.get("logs") or "").splitlines()
            counts[pod["name"]] = (sum(1 for line in lines if ACCESS_LINE.search(line)), len(lines) >= LOG_WINDOW)
    return counts


def _delta(before: dict[str, tuple[int, bool]], after: dict[str, tuple[int, bool]], pods: list[dict[str, Any]]):
    """New requests per pod name, or None when any pod's counts are missing or its log window rolled over."""
    deltas = {}
    for pod in pods:
        if pod["name"] not in before or pod["name"] not in after or before[pod["name"]][1] or after[pod["name"]][1]:
            return None
        deltas[pod["name"]] = after[pod["name"]][0] - before[pod["name"]][0]
    return deltas


def _result(check: str, verdict: str, sufficient: bool, evidence: list[str], limits: list[str] | None = None):
    return {
        "check": check,
        "verdict": verdict,
        "evidence_sufficient": sufficient,
        "evidence": evidence,
        **({"limits": limits} if limits else {}),
    }


def _chat_check(responses: list[tuple[int, dict[str, Any] | None]]) -> tuple[dict[str, Any], str | None]:
    status, body = responses[0]
    choices = (body or {}).get("choices") or []
    if status == 200 and choices:
        message = choices[0].get("message") or {}
        sample = message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""
        return _result("chat", "pass", True, [f"HTTP 200 with {len(choices)} choice(s)"]), str(sample)[:300]
    return _result("chat", "fail", True, [f"HTTP {status} without a completion"]), None


def _routing_check(deltas: dict[str, int] | None, total: int) -> dict[str, Any]:
    if deltas is None:
        return _result(
            "routing_spread",
            "inconclusive",
            False,
            ["Access-log request counts were unavailable or the log window rolled."],
        )
    if len(deltas) < 2:
        return _result(
            "routing_spread", "inconclusive", False, ["Only one serving pod exists; spread cannot be shown."]
        )
    busy = sum(1 for count in deltas.values() if count > 0)
    evidence = [f"{total} requests; per-pod new access-log lines: {sorted(deltas.values(), reverse=True)}"]
    if busy >= 2:
        return _result("routing_spread", "pass", True, evidence)
    if total >= SPREAD_FAIL_MIN_REQUESTS:
        return _result(
            "routing_spread", "fail", True, [*evidence, "all requests landed on one pod although several were ready"]
        )
    return _result("routing_spread", "inconclusive", False, [*evidence, "too few requests to call a failure"])


def _pd_check(guide: str | None, prefill: dict[str, int] | None, decode: dict[str, int] | None) -> dict[str, Any]:
    limits = [
        "Lens exposes no KV-transfer counters; this shows both roles handled the traffic, not the transfer itself."
    ]
    if guide != PD_GUIDE:
        return _result("pd_transfer", "inconclusive", False, ["Not a pd-disaggregation deployment."])
    if prefill is None or decode is None:
        return _result("pd_transfer", "inconclusive", False, ["Access-log request counts were unavailable."], limits)
    prefill_new, decode_new = sum(prefill.values()), sum(decode.values())
    evidence = [f"new requests seen by prefill pods: {prefill_new}; by decode pods: {decode_new}"]
    if prefill_new > 0 and decode_new > 0:
        return _result("pd_transfer", "pass", True, evidence, limits)
    return _result(
        "pd_transfer", "fail", True, [*evidence, "one role saw no requests; prefill/decode split is not working"]
    )


def _prefix_check(responses: list[tuple[int, dict[str, Any] | None]]) -> dict[str, Any]:
    cached = []
    for status, body in responses:
        details = ((body or {}).get("usage") or {}).get("prompt_tokens_details")
        if status == 200 and isinstance(details, dict) and "cached_tokens" in details:
            cached.append(details["cached_tokens"] or 0)
    limits = ["Needs the server to report usage.prompt_tokens_details (vLLM --enable-prompt-tokens-details)."]
    if not cached:
        return _result("prefix_cache", "inconclusive", False, ["Responses carry no cached-token counts."], limits)
    evidence = [f"cached prompt tokens per request after the first: {cached[1:]}"]
    if any(value > 0 for value in cached[1:]):
        return _result("prefix_cache", "pass", True, evidence)
    return _result("prefix_cache", "fail", True, [*evidence, "repeated long prefixes were never served from cache"])


async def run_test(
    ctx: ToolContext, ref: DeploymentRef, view: dict[str, Any], checks: list[str], requests: int
) -> dict[str, Any]:
    execution = view["execution"] or {}
    try:
        forwarded = await ctx.api.request("POST", f"/deployments/executions/{ref.execution_id}/endpoint", timeout=60)
        url = str(forwarded["endpoint"]).rstrip("/")
        _, models = await http_call("GET", f"{url}/v1/models", None)
    except (LensApiError, KeyError, EndpointUnreachableError) as error:
        return _inconclusive(ref, f"The deployment endpoint is not reachable from this server ({_why(error)}).")
    model = ((models or {}).get("data") or [{}])[0].get("id") or execution.get("model")

    wanted = [check for check in CHECKS if check in checks]
    prefix_requests = PREFIX_REQUESTS if "prefix_cache" in wanted else 0
    pool = [
        _chat_payload(model, f"In one sentence, say something about {TOPICS[i % len(TOPICS)]} ({i}).")
        for i in range(max(requests - prefix_requests, 1))
    ]
    pods = view["pods"]
    serving = [p for p in pods if p["role"] in ("decode", "serving", "both")]
    prefill = [p for p in pods if p["role"] == "prefill"]
    watch = serving + prefill if {"routing_spread", "pd_transfer"} & set(wanted) else []

    try:
        before = await _request_counts(ctx, ref.execution_id, watch)
        responses = await _send_all(url, pool)
        after = await _request_counts(ctx, ref.execution_id, watch)
        prefix_responses = []
        if prefix_requests:
            prefix_responses = await _send_all(
                url,
                [_chat_payload(model, f"{SHARED_PREFIX}\nQuestion {i}: name a color.") for i in range(prefix_requests)],
            )
    except EndpointUnreachableError as error:
        return _inconclusive(ref, f"The deployment endpoint stopped answering ({_why(error)}).")

    results, sample = [], None
    for check in wanted:
        if check == "chat":
            result, sample = _chat_check(responses)
        elif check == "routing_spread":
            result = _routing_check(_delta(before, after, serving), len(pool))
        elif check == "pd_transfer":
            result = _pd_check(
                execution.get("guide") or _guide(ref), _delta(before, after, prefill), _delta(before, after, serving)
            )
        elif check == "prefix_cache":
            result = _prefix_check(prefix_responses)
        else:
            result = _result(
                "rank_distribution",
                "inconclusive",
                False,
                ["Lens exposes no per-rank (data-parallel) request counters."],
            )
        results.append(result)
    return _summary(ref, results, len(pool) + len(prefix_responses), sample)


def _guide(ref: DeploymentRef) -> str | None:
    return ((ref.row.meta.get("spec") if ref.row else None) or {}).get("guide")


def _why(error: Exception) -> str:
    return f"Lens HTTP {error.status}" if isinstance(error, LensApiError) else type(error).__name__


def _verdict(results: list[dict[str, Any]]) -> str:
    verdicts = {result["verdict"] for result in results}
    return "fail" if "fail" in verdicts else "pass" if verdicts == {"pass"} else "inconclusive"


def _summary(ref: DeploymentRef, results: list[dict[str, Any]], sent: int, sample: str | None) -> dict[str, Any]:
    return {
        "deployment_id": ref.public_id,
        "verdict": _verdict(results),
        "evidence_sufficient": all(result["evidence_sufficient"] for result in results),
        "checks": results,
        "requests_sent": sent,
        "observed_at": now_iso(),
        "untrusted_text": [{"source": "model_response", "lines": [sample]}] if sample else [],
        "next": "Call get_deployment for readiness, or diagnose_deployment if a check failed.",
    }


def _inconclusive(ref: DeploymentRef, reason: str) -> dict[str, Any]:
    return {
        "deployment_id": ref.public_id,
        "verdict": "inconclusive",
        "evidence_sufficient": False,
        "checks": [],
        "requests_sent": 0,
        "observed_at": now_iso(),
        "reason": reason,
        "next": "The endpoint is forwarded on the Lens host; run the server where it can reach it or use get_logs.",
    }
