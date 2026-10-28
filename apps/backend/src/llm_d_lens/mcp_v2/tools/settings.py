"""Lens-settings profile: read the configured external AI providers and probe their connectivity."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ..context import ToolContext
from ..errors import ToolError
from ..lens_api import LensApiError
from ..registry import tool

PROFILE = "lens-settings"

# Lens stores no test history, so the last result is remembered for the lifetime of this server process.
_LAST_TEST: dict[str, dict[str, Any]] = {}


class ListAiProvidersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbeAiProviderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(description="Provider id from list_ai_providers.")


@tool(
    "list_ai_providers",
    "[R] List the external AI providers configured in Lens: name, model, provider type, endpoint host, whether a "
    "credential is stored (never the credential itself) and the last connectivity test this server ran. "
    "To check that one still works use test_ai_provider.\n"
    "Side effects: none; resources: none; duration: synchronous; reversible: n/a.",
    ListAiProvidersInput,
    read_only=True,
    profile=PROFILE,
)
async def list_ai_providers(ctx: ToolContext, args: ListAiProvidersInput) -> dict[str, Any]:
    payload = await ctx.api.request("GET", "/ai-providers")
    providers = [
        {
            "provider_id": item["id"],
            "name": item.get("name"),
            "model": item.get("model"),
            "provider_type": item.get("providerType"),
            "endpoint_host": urlsplit(str(item.get("baseUrl") or "")).hostname,
            "context_window": item.get("contextWindow"),
            "has_credentials": bool(item.get("hasApiKey")),
            "last_test": _LAST_TEST.get(item["id"]),
        }
        for item in payload.get("items") or []
    ]
    return {"providers": providers, "count": len(providers)}


@tool(
    "test_ai_provider",
    "[W] Send Lens's fixed minimal connectivity request (a model-list call, or one tiny message for providers without "
    "one) to a saved AI provider and report whether it answered. It reaches an external service and may be billed by "
    "that provider. Find ids with list_ai_providers.\n"
    "Side effects: one outbound request using the stored credential; resources: none; duration: synchronous, up to "
    "the provider timeout; reversible: n/a.",
    ProbeAiProviderInput,
    read_only=False,
    open_world=True,
    profile=PROFILE,
)
async def test_ai_provider(ctx: ToolContext, args: ProbeAiProviderInput) -> dict[str, Any]:
    try:
        result = await ctx.api.request("POST", f"/ai-providers/{args.provider_id}/test", timeout=150)
    except LensApiError as error:
        if error.status == 404:
            raise ToolError(
                "not_found",
                f"No AI provider '{args.provider_id}'.",
                field="provider_id",
                hint="Call list_ai_providers.",
            ) from error
        raise
    outcome = {
        "success": bool(result.get("success")),
        "model_found": result.get("modelFound"),
        "latency_ms": result.get("latencyMs"),
        "tested_at": time.time(),
    }
    _LAST_TEST[args.provider_id] = outcome
    return {
        "provider_id": args.provider_id,
        **outcome,
        "untrusted_text": str(result.get("message") or "")[:300],
        "next": "Fix the provider in the Lens UI if the test failed; agents cannot edit providers.",
    }
