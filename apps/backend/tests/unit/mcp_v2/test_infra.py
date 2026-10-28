import pytest

from llm_d_lens.mcp_v2 import kinds
from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.errors import ToolError
from llm_d_lens.mcp_v2.registry import call_tool
from llm_d_lens.mcp_v2.safety import find_secret, redact

from .conftest import FakeApi


@pytest.fixture
def fake_kind(monkeypatch):
    """A kind whose delete handler records what it removed."""
    calls: list[str] = []
    blockers: dict[str, list] = {}
    failures: dict[str, Exception] = {}

    async def delete(ctx, row, cascade):
        if row.id in failures:
            raise failures[row.id]
        calls.append(row.id)

    async def dependents(ctx, row):
        return blockers.get(row.id, [])

    monkeypatch.setitem(kinds._HANDLERS, "model_cache", kinds.KindHandler("model_cache", delete, None, dependents))
    return calls, blockers, failures


def add(ctx: ToolContext, name: str, **kw):
    return ctx.ledger.add_resource(kind="model_cache", owner=ctx.owner, cluster_id="c1", name=name, **kw)


@pytest.mark.asyncio
async def test_delete_reports_each_id_and_only_touches_owned_resources(ctx, fake_kind):
    calls, _, _ = fake_kind
    mine = add(ctx, "qwen")
    theirs = ctx.ledger.add_resource(kind="model_cache", owner="user-b", cluster_id="c1", name="llama")
    payload, is_error = await call_tool(ctx, "delete_resources", {"ids": [mine.id, theirs.id, "mc-00000000", "orph-1"]})
    assert not is_error
    status = {item["id"]: item for item in payload["results"]}
    assert status[mine.id]["status"] == "deleted"
    assert status[theirs.id]["error"] == "not_owner"
    assert status["mc-00000000"]["error"] == "owner_unknown"
    assert status["orph-1"]["error"] == "scope_required"
    assert calls == [mine.id]
    assert ctx.ledger.get(mine.id).state == "deleted" and ctx.ledger.get(theirs.id).state == "active"


@pytest.mark.asyncio
async def test_delete_blocks_in_use_unless_cascade_and_name_mismatch_deletes_nothing(ctx, fake_kind):
    calls, blockers, _ = fake_kind
    row = add(ctx, "qwen")
    blockers[row.id] = [{"id": "dep-1", "reason": "uses this cache"}]
    blocked, _ = await call_tool(ctx, "delete_resources", {"ids": [row.id]})
    assert blocked["results"][0]["status"] == "blocked" and blocked["results"][0]["error"] == "in_use"
    assert calls == [] and ctx.ledger.get(row.id).state == "active"
    cascaded, _ = await call_tool(ctx, "delete_resources", {"ids": [row.id], "cascade": True})
    assert cascaded["deleted"] == [row.id]

    other = add(ctx, "llama")
    mismatch, is_error = await call_tool(ctx, "delete_resources", {"ids": [other.id], "expected_names": ["wrong"]})
    assert is_error and mismatch["error"]["code"] == "id_mismatch" and ctx.ledger.get(other.id).state == "active"


@pytest.mark.asyncio
async def test_delete_restores_state_when_the_handler_fails_and_cascades_children(ctx, fake_kind):
    calls, _, failures = fake_kind
    row = add(ctx, "qwen")
    failures[row.id] = ToolError("in_use", "served by svc-1")
    result, _ = await call_tool(ctx, "delete_resources", {"ids": [row.id]})
    assert result["results"][0]["status"] == "blocked" and ctx.ledger.get(row.id).state == "active"
    del failures[row.id]
    child = add(ctx, "child", root_id=row.id)
    await call_tool(ctx, "delete_resources", {"ids": [row.id]})
    assert ctx.ledger.get(child.id).state == "deleted"


@pytest.mark.asyncio
async def test_protected_resources_are_not_deleted(ctx, fake_kind):
    calls, _, _ = fake_kind
    row = add(ctx, "keep", meta={"protected": True})
    result, _ = await call_tool(ctx, "delete_resources", {"ids": [row.id]})
    assert result["results"][0]["error"] == "protected_resource" and calls == []


@pytest.mark.asyncio
async def test_tools_outside_the_connection_profiles_are_refused(ctx):
    payload, is_error = await call_tool(ctx, "cordon_nodes", {"nodes": ["n1"]})
    assert is_error and payload["error"]["code"] == "scope_required"


@pytest.mark.asyncio
async def test_secrets_are_rejected_in_input_and_redacted_in_output(ctx, fake_kind):
    token = "hf_" + "a" * 30
    payload, is_error = await call_tool(ctx, "delete_resources", {"ids": ["mc-1"], "client_request_id": token})
    assert is_error and payload["error"]["code"] == "secret_in_argument"
    assert find_secret({"env": [{"value": token}]}) == "env[0].value"
    assert redact({"log": f"token {token} end"}) == {"log": "token *** end"}


@pytest.mark.asyncio
async def test_client_request_id_replays_the_first_result_and_destructive_calls_are_rate_limited(
    ctx, fake_kind, monkeypatch
):
    calls, _, _ = fake_kind
    row = add(ctx, "qwen")
    first, _ = await call_tool(ctx, "delete_resources", {"ids": [row.id], "client_request_id": "k1"})
    again, _ = await call_tool(ctx, "delete_resources", {"ids": [row.id], "client_request_id": "k1"})
    assert again["meta"] == {"idempotent_replay": True} and again["deleted"] == first["deleted"] and calls == [row.id]

    monkeypatch.setattr("llm_d_lens.mcp_v2.registry.DESTRUCTIVE_PER_HOUR", 1)
    limited, is_error = await call_tool(ctx, "delete_resources", {"ids": ["mc-00000001"]})
    assert is_error and limited["error"]["code"] == "rate_limited"


@pytest.mark.asyncio
async def test_list_resources_filters_to_the_callers_own_and_shows_dependents(ctx, fake_kind):
    _, blockers, _ = fake_kind
    mine = add(ctx, "qwen")
    ctx.ledger.add_resource(kind="model_cache", owner="user-b", cluster_id="c1", name="llama")
    blockers[mine.id] = [{"id": "dep-1"}]
    payload, _ = await call_tool(ctx, "list_resources", {"kinds": ["model_cache"], "include": ["dependents"]})
    assert [(r["id"], r["ownership"], r["dependents"]) for r in payload["resources"]] == [
        (mine.id, "owned", [{"id": "dep-1"}])
    ]


def test_ledger_ids_use_kind_prefixes(ctx):
    assert add(ctx, "x").id.startswith("mc-")
    assert ctx.ledger.add_resource(kind="experiment", owner="u", cluster_id=None).id.startswith("exp-")


def test_audit_records_writes_without_secrets(ctx):
    ctx.ledger.audit("u", "delete_resources", redact({"note": "hf_" + "b" * 30}), False, None)
    rows = ctx.ledger._read("SELECT arguments FROM audit")
    assert "hf_" not in rows[0]["arguments"]


@pytest.mark.asyncio
async def test_connection_scopes_gate_tools(api: FakeApi):
    from llm_d_lens.mcp_v2.ledger import Ledger

    admin = ToolContext(api=api, ledger=Ledger(":memory:"), owner="u", scopes=frozenset({"core", "cluster-admin"}))
    payload, _ = await call_tool(admin, "cordon_nodes", {"nodes": []})
    assert payload.get("error", {}).get("code") != "scope_required"
