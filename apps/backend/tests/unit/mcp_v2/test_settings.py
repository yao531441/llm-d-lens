import dataclasses

import pytest

from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.registry import call_tool
from llm_d_lens.mcp_v2.tools import settings

PROVIDER = {
    "id": "ai-provider-1a2b3c4d",
    "name": "deepseek",
    "baseUrl": "https://api.deepseek.com/v1?token=abc",
    "model": "deepseek-chat",
    "providerType": "openai",
    "contextWindow": 64000,
    "hasApiKey": True,
    "apiKeyPreview": "sk-…wxyz",
}


@pytest.fixture
def settings_ctx(ctx, monkeypatch):
    monkeypatch.setattr(settings, "_LAST_TEST", {})
    return dataclasses.replace(ctx, scopes=frozenset({"core", "lens-settings"}))


@pytest.mark.asyncio
async def test_profile_is_hidden_without_scope(ctx):
    payload, is_error = await call_tool(ctx, "test_ai_provider", {"provider_id": PROVIDER["id"]})
    assert is_error and payload["error"]["code"] == "scope_required"


@pytest.mark.asyncio
async def test_list_ai_providers_shows_host_and_credential_flag_but_no_secrets(api, settings_ctx):
    api.on("GET", "/ai-providers", {"items": [PROVIDER]})
    payload, is_error = await call_tool(settings_ctx, "list_ai_providers", {})
    assert not is_error, payload
    (provider,) = payload["providers"]
    assert provider["endpoint_host"] == "api.deepseek.com" and provider["has_credentials"] is True
    assert provider["last_test"] is None
    assert "token=abc" not in str(payload) and "sk-" not in str(payload)


@pytest.mark.asyncio
async def test_test_ai_provider_reports_result_and_remembers_it(api, settings_ctx):
    api.on("GET", "/ai-providers", {"items": [PROVIDER]})
    api.on(
        "POST",
        f"/ai-providers/{PROVIDER['id']}/test",
        {"success": True, "message": "Connected successfully.", "modelFound": True, "latencyMs": 120.5},
    )
    payload, is_error = await call_tool(settings_ctx, "test_ai_provider", {"provider_id": PROVIDER["id"]})
    assert not is_error, payload
    assert payload["success"] is True and payload["latency_ms"] == 120.5
    assert payload["untrusted_text"] == "Connected successfully."
    listed, _ = await call_tool(settings_ctx, "list_ai_providers", {})
    assert listed["providers"][0]["last_test"]["success"] is True


@pytest.mark.asyncio
async def test_test_ai_provider_unknown_id_is_not_found(api, settings_ctx):
    api.on(
        "POST", "/ai-providers/nope/test", LensApiError(404, {"detail": "missing"}, "POST", "/ai-providers/nope/test")
    )
    payload, is_error = await call_tool(settings_ctx, "test_ai_provider", {"provider_id": "nope"})
    assert is_error and payload["error"]["code"] == "not_found"
