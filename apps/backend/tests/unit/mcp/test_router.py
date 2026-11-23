import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from llm_d_lens.mcp.lens_api import AsgiLensApi

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


async def rpc(client: httpx.AsyncClient, method: str, params: dict | None = None):
    response = await client.post(
        "/api/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}, headers=HEADERS
    )
    assert response.status_code == 200, response.text
    data = response.text
    if data.startswith("event:"):
        data = next(line[5:] for line in data.splitlines() if line.startswith("data:"))
    import json

    return json.loads(data)["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/guide-planning/plan", "/api/guide-planning/plan"),
        ("/cluster/overview", "/api/cluster/overview"),
        ("/evaluate/evaluations", "/api/v1/evaluate/evaluations"),
    ],
)
async def test_asgi_calls_the_versioned_lens_routes(monkeypatch, path, expected):
    from types import SimpleNamespace

    app = FastAPI()

    @app.post(expected)
    async def receive(request: Request):
        return {"path": request.url.path, "query": request.url.query, "body": await request.json()}

    request = SimpleNamespace(app=app, base_url="http://lens.test")
    asgi = AsgiLensApi(request)
    monkeypatch.setattr(asgi, "_headers", lambda *_: {})
    body = {"model": "Qwen/Qwen3-0.6B"}
    params = {"clusterId": "c1", "enabled": True, "unused": None}
    result = await asgi.request("POST", path, params=params, body=body)

    assert result == {"path": expected, "query": "clusterId=c1&enabled=true", "body": body}
