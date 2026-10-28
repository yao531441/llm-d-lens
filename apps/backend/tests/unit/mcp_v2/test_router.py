import httpx
import pytest
from fastapi import FastAPI

from llm_d_lens.mcp_v2 import router as router_module
from llm_d_lens.mcp_v2.ledger import Ledger

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


async def rpc(client: httpx.AsyncClient, method: str, params: dict | None = None):
    response = await client.post(
        "/api/mcp/v2", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}, headers=HEADERS
    )
    assert response.status_code == 200, response.text
    data = response.text
    if data.startswith("event:"):
        data = next(line[5:] for line in data.splitlines() if line.startswith("data:"))
    import json

    return json.loads(data)["result"]


@pytest.mark.asyncio
async def test_streamable_http_lists_tools_with_annotations_and_returns_structured_errors(monkeypatch):
    monkeypatch.setattr(router_module, "_ledger", Ledger(":memory:"))
    app = FastAPI()
    app.include_router(router_module.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        listed = await rpc(client, "tools/list")
        tools = {tool["name"]: tool for tool in listed["tools"]}
        assert {"plan_deployment", "create_deployment", "delete_deployment"} <= set(tools)
        assert tools["delete_deployment"]["annotations"]["destructiveHint"] is True
        assert tools["get_deployment"]["annotations"]["readOnlyHint"] is True

        called = await rpc(client, "tools/call", {"name": "create_deployment", "arguments": {}})
        assert called["isError"] is True
        assert "not authenticated" in called["content"][0]["text"]

        denied = await client.get("/api/mcp/v2")
        assert denied.status_code == 405
