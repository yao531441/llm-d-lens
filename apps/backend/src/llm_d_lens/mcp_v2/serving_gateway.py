"""Inference requests through the managed model gateway for ``test_model_service``."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

TOKEN_ENV = "LENS_MCP_V2_GATEWAY_TOKEN"  # noqa: S105 - env var name, not a secret
MAX_REPLY_CHARS = 300


def gateway_token() -> str | None:
    """A service token the human created in the Lens UI and put in this server's environment."""
    return os.environ.get(TOKEN_ENV) or None


async def send_chat(base_url: str, token: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            )
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"[:200], "latency_ms": _elapsed(started)}
    result: dict[str, Any] = {"ok": response.status_code == 200, "http_status": response.status_code}
    result["latency_ms"] = _elapsed(started)
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code == 200:
        choice = (body.get("choices") or [{}])[0]
        result["finish_reason"] = choice.get("finish_reason")
        result["untrusted_text"] = str((choice.get("message") or {}).get("content") or "")[:MAX_REPLY_CHARS]
    else:
        detail = body.get("detail") or body.get("error") or response.text
        result["untrusted_text"] = str(detail)[:MAX_REPLY_CHARS]
    return result


def _elapsed(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
