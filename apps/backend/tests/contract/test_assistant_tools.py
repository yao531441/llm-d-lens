import pytest

from llm_d_lens.assistant_tools.catalog import generate_catalog, render_path


def test_generated_catalog_keeps_names_parameters_and_risk_tiers():
    spec = {
        "paths": {
            "/api/example/{id}": {
                "post": {
                    "operationId": "approve_agentic_plan",
                    "summary": "Approve",
                    "description": "Apply the selected plan.",
                    "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"count": {"type": "integer"}},
                                    "required": ["count"],
                                }
                            }
                        }
                    },
                }
            }
        }
    }
    tool = generate_catalog(spec)[-1]
    assert tool["name"] == "approve_agentic_plan"
    assert tool["riskTier"] == "approve"
    assert tool["bodyFields"] == ["count"]
    assert tool["inputSchema"]["required"] == ["id", "count"]
    assert render_path(tool["route"], {"id": "part/with space"}) == "/api/example/part%2Fwith%20space"


def test_publish_tools_distinguish_deployment_readiness_from_model_service():
    from llm_d_lens.api.main import app
    from llm_d_lens.assistant_tools.catalog import catalog

    tools = {tool["name"]: tool for tool in catalog(app)}
    assert "publishing step, not deployment" in tools["publish_model_service"]["description"].lower()
    assert "does not publish anything" in tools["list_publishable_deployments"]["description"].lower()
    assert "existing public model service" in tools["create_model_service_member"]["description"].lower()
    assert "does not mean the deployment was published" in tools["list_ready_deployments"]["description"].lower()


def test_generator_refuses_unclassified_operations():
    with pytest.raises(ValueError, match="Unexpected MCP"):
        generate_catalog({"paths": {"/new-route": {"post": {"operationId": "new_route_post"}}}})


def test_full_backend_catalog_has_unique_names_and_special_risk_tiers():
    from llm_d_lens.api.main import app

    tools = generate_catalog(app.openapi())
    assert len(tools) > 100
    assert len({tool["name"] for tool in tools}) == len(tools)
    indexed = {tool["name"]: tool for tool in tools}
    assert indexed["start_remote_deploy"]["riskTier"] == "approve"
    assert indexed["wait_for_status"]["riskTier"] == "read"
    assert indexed["probe_model_gateway"]["riskTier"] == "read"


@pytest.mark.asyncio
async def test_wait_returns_pending_without_sleeping():
    from unittest.mock import AsyncMock

    from llm_d_lens.assistant_tools.tools import poll_until_terminal_or_pending

    poll = AsyncMock(return_value={"pods": [{"phase": "Pending"}]})
    result = await poll_until_terminal_or_pending(poll, 600, 30)
    assert result["pending"] is True
    assert result["done"] is False
    poll.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_cannot_invoke_approve_tool():
    from types import SimpleNamespace

    from llm_d_lens.api.main import app
    from llm_d_lens.assistant_tools.tools import invoke_tool

    result = await invoke_tool(SimpleNamespace(app=app), "wait_for_status", {"toolName": "start_remote_deploy"})
    assert "only poll read-tier tools" in result["error"]


@pytest.mark.asyncio
async def test_storage_tool_preserves_repeated_query_parameters(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from llm_d_lens.api.main import app
    from llm_d_lens.assistant_tools import tools

    upstream = AsyncMock(return_value={"items": []})
    monkeypatch.setattr(tools, "api_request", upstream)
    await tools.invoke_tool(SimpleNamespace(app=app), "get_storage_volume_resource_status", {"volumeIds": ["a", "b"]})
    assert upstream.call_args.args[1].endswith("?volume_id=a&volume_id=b")


def mcp_request(client, method, params=None):
    import json

    from httpx_sse import EventSource

    response = client.post(
        "/api/assistant/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers={"accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200, response.text
    if response.headers["content-type"].startswith("text/event-stream"):
        return json.loads(next(EventSource(response).iter_sse()).data)
    return response.json()


def test_mcp_http_initialization_list_and_read_tool():
    import json

    from fastapi.testclient import TestClient

    from llm_d_lens.api.main import app

    with TestClient(app) as client:
        initialized = mcp_request(
            client,
            "initialize",
            {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
        )
        assert initialized["result"]["serverInfo"]["name"] == "llm-d-prism"
        listed = mcp_request(client, "tools/list")
        tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
        assert tools["start_remote_deploy"]["description"].startswith("[approve]")
        called = mcp_request(client, "tools/call", {"name": "get_remote_deploy_defaults", "arguments": {}})
        assert not called["result"].get("isError")
        assert json.loads(called["result"]["content"][0]["text"])
        assert client.get("/api/assistant/mcp").status_code == 405
        assert client.delete("/api/assistant/mcp").status_code == 405


@pytest.mark.asyncio
async def test_validation_errors_do_not_echo_argument_values():
    from types import SimpleNamespace

    from llm_d_lens.api.main import app
    from llm_d_lens.assistant_tools.tools import invoke_tool

    with pytest.raises(ValueError) as caught:
        await invoke_tool(
            SimpleNamespace(app=app), "get_deployment_execution", {"execution_id": {"private": "sensitive-value"}}
        )
    assert "execution_id" in str(caught.value)
    assert "sensitive-value" not in str(caught.value)


@pytest.mark.asyncio
async def test_agentic_client_connects_to_python_mcp_and_calls_the_canonical_api(monkeypatch):
    from functools import partial
    from types import SimpleNamespace

    import httpx

    from llm_d_lens.agentic import generator
    from llm_d_lens.api.main import app
    from llm_d_lens.assistant_tools.catalog import catalog

    monkeypatch.delenv("PRISM_MCP_CA_FILE", raising=False)
    transport = httpx.ASGITransport(app=app)
    monkeypatch.setattr(
        generator, "httpx", SimpleNamespace(AsyncClient=partial(httpx.AsyncClient, transport=transport))
    )
    client = generator.AgenticMcpClient("http://testserver/api/assistant/mcp")
    available = await client.available_tools()
    assert {"get_cluster_overview", "search_candidates"} <= {tool["name"] for tool in available}
    result = await client.call("get_configuration_capabilities", {})
    definition = next(tool for tool in catalog(app) if tool["name"] == "get_configuration_capabilities")
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as canonical:
        expected = await canonical.get(definition["route"])
    assert expected.status_code == 200
    assert result == expected.json()


@pytest.mark.parametrize("field", ["concurrency", "request_rate"])
@pytest.mark.asyncio
async def test_candidate_tools_advertise_and_forward_load_targets(monkeypatch, field):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from jsonschema import Draft202012Validator

    from llm_d_lens.api.main import app
    from llm_d_lens.assistant_tools import tools
    from llm_d_lens.assistant_tools.catalog import catalog

    indexed = {tool["name"]: tool for tool in catalog(app)}
    for name in ("check_candidate_support", "search_candidates", "search_aic_with_relaxation"):
        properties = indexed[name]["inputSchema"]["properties"]["workload"]["properties"]
        assert properties[field]["exclusiveMinimum"] == 0
        assert not Draft202012Validator(properties[field]).is_valid(0)
    upstream = AsyncMock(return_value={"candidates": []})
    monkeypatch.setattr(tools, "api_request", upstream)
    workload = {"model": "model", "isl": 4000, "osl": 1000, field: 10}
    await tools.invoke_tool(
        SimpleNamespace(app=app), "search_candidates", {"workload": workload, "searchConfig": {}, "sourceIds": ["aic"]}
    )
    assert upstream.call_args.args[3]["workload"] == workload
