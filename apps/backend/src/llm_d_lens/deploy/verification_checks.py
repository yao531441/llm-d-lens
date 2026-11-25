"""Bounded deployment smoke checks, independent of HTTP routes and MCP."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

Check = Literal["chat", "prefix_cache"]
PREFIX_REQUESTS = 6


class VerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[Check] = Field(default_factory=lambda: ["chat"], min_length=1, max_length=2)
    requests: int = Field(default=10, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def validate_budget(self):
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks must be unique")
        if "prefix_cache" in self.checks and self.requests < 10:
            raise ValueError("prefix_cache requires requests >= 10, including six prefix requests")
        return self


async def http_call(url: str, payload: dict) -> tuple[int, dict | None]:
    """Use only the Lens-resolved local endpoint; never follow model-server redirects."""
    async with (
        httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client,
        client.stream("POST", url, json=payload) as response,
    ):
        data = bytearray()
        async for chunk in response.aiter_bytes():
            data.extend(chunk)
            if len(data) > 65536:
                return response.status_code, None
        try:
            body = json.loads(data)
            return response.status_code, body if isinstance(body, dict) else None
        except ValueError:
            return response.status_code, None


def result(check: str, verdict: str, sufficient: bool, evidence: str) -> dict:
    return {"check": check, "verdict": verdict, "evidence_sufficient": sufficient, "evidence": [evidence]}


async def run_checks(
    endpoint: str, model: str, request: VerificationRequest, progress: Callable[..., None] | None = None
) -> dict:
    """Send no more than the total budget, warming repeated prefixes sequentially.

    ``progress`` receives the request counters after every request, so an interrupted run keeps what was sent.
    """
    sent = completed = succeeded = output_tokens = 0
    prefix_responses: list[tuple[int, dict | None]] = []
    prefix_count = PREFIX_REQUESTS if "prefix_cache" in request.checks else 0
    shared = " ".join(f"Rule {i}: answer briefly and stay on topic." for i in range(60))
    for index in range(request.requests):
        prefix = index >= request.requests - prefix_count
        prompt = f"{shared}\nName a color." if prefix else f"Name a river in one sentence. Request {index}."
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
        }
        sent += 1
        try:
            status, body = await http_call(f"{endpoint.rstrip('/')}/v1/chat/completions", payload)
        except httpx.HTTPError:
            status, body = 0, None
        completed += int(status != 0)
        choices = (body or {}).get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        valid = (
            status == 200
            and isinstance(message, dict)
            and any(
                isinstance(message.get(key), str) and bool(message[key])
                for key in ("content", "reasoning_content", "reasoning")
            )
        )
        succeeded += int(valid)
        usage = (body or {}).get("usage")
        count = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            output_tokens += count
        if prefix:
            prefix_responses.append((status, body))
        if progress:
            progress(
                requests_sent=sent,
                requests_completed=completed,
                requests_succeeded=succeeded,
                output_tokens=output_tokens,
            )

    checks = []
    for check in request.checks:
        if check == "chat":
            verdict = "fail" if succeeded < completed else "pass" if succeeded == sent else "inconclusive"
            checks.append(
                result(
                    check,
                    verdict,
                    verdict != "inconclusive",
                    f"{succeeded}/{sent} requests returned completions; "
                    f"{completed} HTTP responses received. This verifies response availability only.",
                )
            )
        elif check == "prefix_cache":
            cached = []
            for status, body in prefix_responses[1:]:
                usage = (body or {}).get("usage")
                details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
                count = details.get("cached_tokens") if isinstance(details, dict) else None
                if status == 200 and isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    cached.append(count)
            checks.append(
                result(
                    check,
                    "pass" if cached else "inconclusive",
                    bool(cached),
                    "Server reported cached prompt tokens on repeated requests."
                    if cached
                    else "No positive cached-token evidence; absent or zero decode counters do not prove a cache miss.",
                )
            )
    verdicts = {item["verdict"] for item in checks}
    return {
        "requests_sent": sent,
        "requests_completed": completed,
        "requests_succeeded": succeeded,
        "output_tokens": output_tokens,
        "checks": checks,
        "verdict": "fail" if "fail" in verdicts else "pass" if verdicts == {"pass"} else "inconclusive",
        "evidence_sufficient": all(item["evidence_sufficient"] for item in checks),
    }
