import pytest

from llm_d_lens.mcp import kinds, tools  # noqa: F401 - registers the real kinds before the fake replaces one
from llm_d_lens.mcp.context import ToolContext
from llm_d_lens.mcp.errors import ToolError
from llm_d_lens.mcp.registry import call_tool
from llm_d_lens.mcp.safety import find_secret, redact

from .conftest import FakeApi


@pytest.fixture
def fake_kind(monkeypatch):
    """A deployment kind backed by a dict, so the generic tools can be tested without Lens."""
    store: dict[str, dict] = {}
    deleted: list[str] = []
    failures: dict[str, Exception] = {}

    async def list_(ctx, cluster_id):
        return list(store.values())

    async def get(ctx, resource_id):
        return store.get(resource_id)

    async def delete(ctx, resource_id):
        if resource_id in failures:
            raise failures[resource_id]
        deleted.append(resource_id)
        return [{"kind": "deployment", "id": "child"}]

    monkeypatch.setitem(kinds._HANDLERS, "deployment", kinds.KindHandler("deployment", list_, get, delete))
    return store, deleted, failures


def add(store: dict, resource_id: str, name: str, ownership: str = "owned") -> None:
    store[resource_id] = {"id": resource_id, "kind": "deployment", "name": name, "ownership": ownership}


def item(resource_id: str, name: str) -> dict:
    return {"kind": "deployment", "id": resource_id, "name": name}


@pytest.mark.asyncio
async def test_delete_reports_each_item_and_only_touches_owned_resources(ctx, fake_kind):
    store, deleted, _ = fake_kind
    add(store, "a", "qwen")
    add(store, "b", "llama", ownership="shared")
    add(store, "c", "mystery", ownership="unknown")
    items = [item("a", "qwen"), item("b", "llama"), item("c", "mystery"), item("gone", "x")]
    payload, is_error = await call_tool(ctx, "resource_delete", {"items": items})
    assert not is_error
    status = {result["id"]: result for result in payload["results"]}
    assert status["a"]["status"] == "deleted" and status["a"]["also_deletes"] == [{"kind": "deployment", "id": "child"}]
    assert status["b"]["error"] == "not_owner"
    assert status["c"]["error"] == "owner_unknown"
    assert status["gone"]["status"] == "not_found"
    assert deleted == ["a"]
    assert payload["summary"] == {"deleted": 1, "blocked": 0, "not_found": 1, "failed": 2}


@pytest.mark.asyncio
async def test_a_name_mismatch_deletes_nothing(ctx, fake_kind):
    store, deleted, _ = fake_kind
    add(store, "a", "qwen")
    add(store, "b", "llama")
    payload, is_error = await call_tool(ctx, "resource_delete", {"items": [item("a", "qwen"), item("b", "wrong")]})
    assert is_error and payload["error"]["code"] == "id_mismatch" and deleted == []


@pytest.mark.asyncio
async def test_in_use_resources_are_blocked_and_unexpected_timeouts_are_reported_per_item(ctx, fake_kind):
    store, deleted, failures = fake_kind
    add(store, "a", "qwen")
    add(store, "b", "llama")
    failures["a"] = ToolError("in_use", "served by a model service")
    failures["b"] = TimeoutError("slow")
    payload, is_error = await call_tool(ctx, "resource_delete", {"items": [item("a", "qwen"), item("b", "llama")]})
    assert not is_error and deleted == []
    assert [r["status"] for r in payload["results"]] == ["blocked", "failed"]
    assert payload["results"][1]["error"] == "timeout"


@pytest.mark.asyncio
async def test_tools_outside_the_connection_profiles_are_refused(ctx):
    payload, is_error = await call_tool(ctx, "orphan_resource_list", {"cluster_id": "b9d6d401"})
    assert is_error and payload["error"]["code"] == "scope_required"


@pytest.mark.asyncio
async def test_secrets_are_rejected_in_input_and_redacted_in_output(ctx, fake_kind):
    token = "hf_" + "a" * 30
    payload, is_error = await call_tool(ctx, "resource_delete", {"items": [item("a", token)]})
    assert is_error and payload["error"]["code"] == "secret_in_argument"
    assert find_secret({"env": [{"value": token}]}) == "env[0].value"
    assert redact({"log": f"token {token} end"}) == {"log": "token *** end"}


@pytest.mark.asyncio
async def test_list_resources_keeps_the_callers_own_unless_owner_is_all(ctx, fake_kind):
    store, _, _ = fake_kind
    add(store, "a", "qwen")
    add(store, "b", "llama", ownership="shared")
    mine, _ = await call_tool(ctx, "resource_list", {"kinds": ["deployment"]})
    assert [r["id"] for r in mine["resources"]] == ["a"]
    everything, _ = await call_tool(ctx, "resource_list", {"kinds": ["deployment"], "owner": "all"})
    assert [r["id"] for r in everything["resources"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_connection_scopes_gate_tools(api: FakeApi):
    admin = ToolContext(api=api, owner="u", scopes=frozenset({"core", "cluster-admin"}))
    payload, _ = await call_tool(admin, "orphan_resource_list", {})
    assert payload.get("error", {}).get("code") != "scope_required"
