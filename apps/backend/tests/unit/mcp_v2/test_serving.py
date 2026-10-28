import pytest

from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.ledger import Ledger
from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.registry import all_tools, call_tool

from .conftest import FakeApi

ADMIN = "/model-service/admin"
SCOPES = frozenset({"core", "serving", "storage"})


class Gateway:
    """A tiny stateful stand-in for the Lens model-service admin API."""

    def __init__(self, api: FakeApi):
        self.api = api
        self.group: dict | None = None
        self.members: list[dict] = []
        self.grants: list[dict] = []
        self.revision = 0
        self.publishable = [
            {"executionId": "exec-1", "name": "qwen-a", "model": "Qwen/Qwen3-0.6B", "clusterId": "c1"},
            {"executionId": "exec-2", "name": "qwen-b", "model": "Qwen/Qwen3-0.6B", "clusterId": "c1"},
        ]
        api.on("GET", f"{ADMIN}/deployments", lambda p, b: {"items": self.publishable})
        api.on("GET", f"{ADMIN}/groups", lambda p, b: {"items": [self.group] if self.group else []})
        api.on("POST", f"{ADMIN}/groups/publish", self.publish)
        api.on("GET", f"{ADMIN}/groups/msg-1", lambda p, b: self.group)
        api.on("GET", f"{ADMIN}/members", lambda p, b: {"items": self.members})
        api.on("POST", f"{ADMIN}/members", self.add_member)
        api.on("PUT", f"{ADMIN}/groups/msg-1/traffic", self.traffic)
        api.on("PATCH", f"{ADMIN}/groups/msg-1", self.patch)
        api.on("POST", f"{ADMIN}/groups/msg-1/status", self.status)
        api.on("DELETE", f"{ADMIN}/groups/msg-1", self.delete_group)
        api.on("GET", f"{ADMIN}/groups/msg-1/access", lambda p, b: {"items": self.grants})
        api.on(
            "GET",
            f"{ADMIN}/groups/msg-1/access/options",
            {"users": [{"id": "u-bob", "username": "bob"}], "groups": [{"id": "g-ml", "name": "ml"}]},
        )
        api.on("POST", f"{ADMIN}/groups/msg-1/access", self.grant)
        api.on(
            "GET",
            "/model-service/connection",
            {"clusters": [{"clusterId": "c1", "baseUrl": "https://gw/v1", "ready": True}]},
        )
        api.on("GET", "/model-service/models", lambda p, b: {"items": [{"id": "msg-1"}] if self.group else []})
        api.on("GET", "/model-service/tokens", {"items": [{"id": "tok-1", "name": "ci", "createdAt": "t0"}]})
        api.on("DELETE", "/model-service/tokens/tok-1", lambda p, b: None)

    def bump(self):
        self.revision += 1
        self.group["updatedAt"] = f"t{self.revision}"

    def member(self, execution_id: str, weight: int = 1) -> dict:
        row = {
            "id": f"mem-{execution_id}",
            "executionId": execution_id,
            "status": "active",
            "weight": weight,
            "healthJson": {"ready": True},
            "name": execution_id,
        }
        self.api.on("DELETE", f"{ADMIN}/members/{row['id']}", lambda p, b, row=row: self.members.remove(row))
        return row

    def publish(self, params, body):
        self.group = {
            "id": "msg-1",
            "name": body["name"],
            "clusterId": body["clusterId"],
            "status": "active",
            "selectionPolicy": "random",
            "description": body["description"],
            "updatedAt": "t0",
        }
        self.members = [self.member(body["executionId"])]
        return self.group

    def add_member(self, params, body):
        self.members.append(self.member(body["executionId"]))
        self.bump()
        return {}

    def traffic(self, params, body):
        current = {m["executionId"]: m["weight"] for m in self.members}
        if body["expectedWeights"] != current:
            raise LensApiError(409, {"code": "model_traffic_conflict", "detail": "expected weights changed"}, "PUT", "")
        assert {m["executionId"] for m in body["members"]} >= set(current)
        for item in body["members"]:
            existing = next((m for m in self.members if m["executionId"] == item["executionId"]), None)
            if existing is None:
                existing = self.member(item["executionId"])
                self.members.append(existing)
            existing["weight"] = item["weight"]
        self.bump()
        return {"items": self.members}

    def patch(self, params, body):
        self.group.update(body)
        self.bump()
        return self.group

    def status(self, params, body):
        self.group["status"] = body["status"]
        self.bump()
        return self.group

    def delete_group(self, params, body):
        assert not self.members, "members must be detached first"
        self.group = None

    def grant(self, params, body):
        if any(g["subjectId"] == body["subjectId"] for g in self.grants):
            raise LensApiError(409, {"detail": "already granted"}, "POST", "")
        grant = {"id": f"bind-{len(self.grants)}", "subjectType": body["subjectType"], "subjectId": body["subjectId"]}
        grant["subjectName"] = {"u-bob": "bob", "g-ml": "ml"}[body["subjectId"]]
        self.grants.append(grant)
        self.api.on("DELETE", f"{ADMIN}/groups/msg-1/access/{grant['id']}", lambda p, b: self.grants.remove(grant))
        return grant


@pytest.fixture
def ctx(api: FakeApi) -> ToolContext:
    return ToolContext(api=api, ledger=Ledger(":memory:"), owner=api.owner, scopes=SCOPES)


@pytest.fixture
def lens(api: FakeApi) -> Gateway:
    return Gateway(api)


def add_deployment(ctx: ToolContext, execution_id: str, owner: str | None = None) -> str:
    return ctx.ledger.add_resource(
        kind="deployment", owner=owner or ctx.owner, cluster_id="c1", name=execution_id, lens_execution_id=execution_id
    ).id


async def publish(ctx: ToolContext, **extra) -> dict:
    dep = add_deployment(ctx, "exec-1")
    payload, is_error = await call_tool(
        ctx, "publish_model_service", {"public_name": "qwen", "deployment_ids": [dep], **extra}
    )
    assert not is_error, payload
    return payload


def test_serving_tools_declare_annotations_from_the_spec():
    expected = {
        "publish_model_service": (False, False, True, True),
        "get_model_service": (True, False, True, False),
        "test_model_service": (False, False, False, False),
        "update_model_service": (False, True, True, False),
        "enable_model_service": (False, False, True, False),
        "disable_model_service": (False, True, True, False),
        "grant_model_service_access": (False, False, True, False),
        "revoke_model_service_access": (False, True, True, False),
    }
    tools = {spec.name: spec for spec in all_tools()}
    for name, flags in expected.items():
        spec = tools[name]
        assert (spec.read_only, spec.destructive, spec.idempotent, spec.open_world) == flags, name
        assert spec.profile == "serving"
        assert spec.description.startswith(("[R]", "[W]", "[S]", "[D]")) and "Side effects:" in spec.description


@pytest.mark.asyncio
async def test_serving_tools_require_the_profile(api, lens):
    ctx = ToolContext(api=api, ledger=Ledger(":memory:"), owner="user-a")
    payload, is_error = await call_tool(ctx, "get_model_service", {"service_id": "svc-x"})
    assert is_error and payload["error"]["code"] == "scope_required"


@pytest.mark.asyncio
async def test_publish_records_ledger_returns_endpoint_without_token(api, ctx, lens):
    payload = await publish(ctx, description="demo")
    row = ctx.ledger.get(payload["service_id"])
    assert row.kind == "model_service" and row.state == "active" and row.lens_execution_id == "msg-1"
    assert payload["endpoint"]["base_url"] == "https://gw/v1" and payload["endpoint"]["model"] == "qwen"
    assert payload["access"] == "owner only" and payload["is_mine"] and payload["status"] == "active"
    assert payload["members"][0]["deployment_id"].startswith("dep-") and payload["members"][0]["owned"]
    assert "token" not in " ".join(k for k in payload if k != "next").lower()
    assert api.called("POST", f"{ADMIN}/groups/publish")[0]["executionId"] == "exec-1"


@pytest.mark.asyncio
async def test_publish_name_conflict_is_already_exists_and_creates_nothing(api, ctx, lens):
    await publish(ctx)
    dep = add_deployment(ctx, "exec-2")
    payload, is_error = await call_tool(ctx, "publish_model_service", {"public_name": "qwen", "deployment_ids": [dep]})
    assert is_error and payload["error"]["code"] == "already_exists" and payload["error"]["field"] == "public_name"
    assert len(api.called("POST", f"{ADMIN}/groups/publish")) == 1
    assert len(ctx.ledger.list_resources(owner="user-a", kind="model_service")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("foreign", "code"), [(False, "owner_unknown"), (True, "not_owner")])
async def test_publish_refuses_deployments_that_are_not_yours(ctx, lens, foreign, code):
    if foreign:
        add_deployment(ctx, "exec-2", owner="bob")
    payload, is_error = await call_tool(
        ctx, "publish_model_service", {"public_name": "x1", "deployment_ids": ["exec-2"]}
    )
    assert is_error and payload["error"]["code"] == code
    assert lens.group is None


@pytest.mark.asyncio
async def test_publish_requires_a_ready_llmd_deployment(ctx, lens):
    lens.publishable = []
    dep = add_deployment(ctx, "exec-1")
    payload, is_error = await call_tool(ctx, "publish_model_service", {"public_name": "x1", "deployment_ids": [dep]})
    assert is_error and payload["error"]["code"] == "conflict_state"
    assert ctx.ledger.list_resources(owner="user-a", kind="model_service") == []


@pytest.mark.asyncio
async def test_publish_two_members_with_weights_applies_traffic(api, ctx, lens):
    deps = [add_deployment(ctx, "exec-1"), add_deployment(ctx, "exec-2")]
    payload, is_error = await call_tool(
        ctx,
        "publish_model_service",
        {"public_name": "qwen", "deployment_ids": deps, "weights": {deps[0]: 90, deps[1]: 10}},
    )
    assert not is_error, payload
    assert {m["deployment_id"]: m["weight"] for m in payload["members"]} == {deps[0]: 90, deps[1]: 10}
    assert api.called("PUT", f"{ADMIN}/groups/msg-1/traffic")[0]["expectedWeights"] == {"exec-1": 1, "exec-2": 1}


@pytest.mark.asyncio
async def test_failed_member_add_rolls_back_the_partial_service(api, ctx, lens):
    api.on("POST", f"{ADMIN}/members", LensApiError(409, {"detail": "plain vLLM is not supported"}, "POST", ""))
    deps = [add_deployment(ctx, "exec-1"), add_deployment(ctx, "exec-2")]
    payload, is_error = await call_tool(ctx, "publish_model_service", {"public_name": "qwen", "deployment_ids": deps})
    assert is_error and payload["error"]["code"] == "unsupported_combination"
    assert lens.group is None
    assert ctx.ledger.list_resources(owner="user-a", kind="model_service") == []


@pytest.mark.asyncio
async def test_get_includes_sections_for_owner_and_hides_them_from_others(api, ctx, lens):
    service = (await publish(ctx))["service_id"]
    lens.grants.append({"id": "b0", "subjectType": "user", "subjectId": "u-bob", "subjectName": "bob"})
    api.on("GET", f"{ADMIN}/usage/analytics", {"totals": {"requests": 3, "inputTokens": 10, "outputTokens": 20}})
    payload, is_error = await call_tool(
        ctx,
        "get_model_service",
        {"service_id": service, "include": ["access", "access_options", "usage", "connection"]},
    )
    assert not is_error, payload
    assert payload["access"]["grants"][0]["subject"] == "user:bob"
    assert payload["access"]["your_tokens"] == [
        {"token_id": "tok-1", "name": "ci", "created_at": "t0", "last_used_at": None, "expires_at": None}
    ]
    assert payload["access_options"]["users"] == ["user:bob"]
    assert payload["usage"]["requests"] == 3 and "curl" in payload["connection"]["curl"]

    other = ToolContext(api=api, ledger=Ledger(":memory:"), owner="user-b", scopes=SCOPES)
    payload, is_error = await call_tool(other, "get_model_service", {"service_id": "msg-1", "include": ["access"]})
    assert not is_error and not payload["is_mine"] and "access" not in payload and "permission_note" in payload


@pytest.mark.asyncio
async def test_update_merges_weights_adds_member_and_returns_full_table(api, ctx, lens):
    published = await publish(ctx)
    service = published["service_id"]
    dep2 = add_deployment(ctx, "exec-2")
    payload, is_error = await call_tool(
        ctx,
        "update_model_service",
        {
            "service_id": service,
            "add_members": [{"deployment_id": dep2, "weight": 25}],
            "weights": {published["members"][0]["deployment_id"]: 75},
            "selection_policy": "weighted",
            "description": "canary",
            "expected_version": published["version"],
        },
    )
    assert not is_error, payload
    assert {m["weight"] for m in payload["members"]} == {75, 25}
    assert payload["selection_policy"] == "weighted" and payload["description"] == "canary"
    assert payload["changes"]["added"] == [dep2] and payload["version"] != published["version"]
    assert api.called("PATCH", f"{ADMIN}/groups/msg-1") == [{"selectionPolicy": "weighted", "description": "canary"}]


@pytest.mark.asyncio
async def test_update_with_stale_version_is_version_conflict_and_changes_nothing(api, ctx, lens):
    published = await publish(ctx)
    payload, is_error = await call_tool(
        ctx,
        "update_model_service",
        {"service_id": published["service_id"], "description": "x", "expected_version": "v-stale"},
    )
    assert is_error and payload["error"]["code"] == "version_conflict"
    assert payload["error"]["current_version"] == published["version"]
    assert api.called("PATCH", f"{ADMIN}/groups/msg-1") == []


@pytest.mark.asyncio
async def test_update_refuses_to_leave_no_traffic(ctx, lens):
    published = await publish(ctx)
    service, member = published["service_id"], published["members"][0]["deployment_id"]
    payload, is_error = await call_tool(
        ctx, "update_model_service", {"service_id": service, "remove_members": [member]}
    )
    assert is_error and payload["error"]["code"] == "in_use"
    payload, is_error = await call_tool(ctx, "update_model_service", {"service_id": service, "weights": {member: 0}})
    assert is_error and payload["error"]["code"] == "invalid_argument"
    assert lens.members[0]["weight"] == 1


@pytest.mark.asyncio
async def test_update_removes_member_after_shifting_traffic(api, ctx, lens):
    deps = [add_deployment(ctx, "exec-1"), add_deployment(ctx, "exec-2")]
    published, _ = await call_tool(ctx, "publish_model_service", {"public_name": "qwen", "deployment_ids": deps})
    payload, is_error = await call_tool(
        ctx, "update_model_service", {"service_id": published["service_id"], "remove_members": [deps[1]]}
    )
    assert not is_error, payload
    assert [m["deployment_id"] for m in payload["members"]] == [deps[0]]
    assert payload["changes"]["removed"] == [deps[1]]


@pytest.mark.asyncio
async def test_writes_on_services_not_in_the_ledger_are_owner_unknown(api, ctx, lens):
    await publish(ctx)
    other = ToolContext(api=api, ledger=Ledger(":memory:"), owner="user-b", scopes=SCOPES)
    for name, args in [
        ("update_model_service", {"description": "x"}),
        ("enable_model_service", {}),
        ("disable_model_service", {}),
        ("grant_model_service_access", {"subjects": ["user:bob"]}),
        ("test_model_service", {}),
    ]:
        payload, is_error = await call_tool(other, name, {"service_id": "msg-1", **args})
        assert is_error and payload["error"]["code"] == "owner_unknown", name
    assert lens.group["status"] == "active" and lens.group["description"] == ""


@pytest.mark.asyncio
async def test_disable_then_enable(ctx, lens):
    service = (await publish(ctx))["service_id"]
    payload, _ = await call_tool(ctx, "disable_model_service", {"service_id": service})
    assert payload["status"] == "disabled" and payload["enabled"] is False
    payload, _ = await call_tool(ctx, "enable_model_service", {"service_id": service})
    assert payload["enabled"] is True and payload["status"] == "active"


@pytest.mark.asyncio
async def test_saved_but_not_applied_becomes_a_warning(api, ctx, lens):
    service = (await publish(ctx))["service_id"]
    api.on(
        "POST",
        f"{ADMIN}/groups/msg-1/status",
        LensApiError(502, {"code": "model_publication_apply_failed", "detail": "route not applied"}, "POST", ""),
    )
    payload, is_error = await call_tool(ctx, "disable_model_service", {"service_id": service})
    assert not is_error and payload["warnings"]


@pytest.mark.asyncio
async def test_grant_and_revoke_access(api, ctx, lens):
    service = (await publish(ctx))["service_id"]
    payload, is_error = await call_tool(
        ctx, "grant_model_service_access", {"service_id": service, "subjects": ["user:bob", "group:ml", "user:eve"]}
    )
    assert not is_error, payload
    assert [r["status"] for r in payload["results"]] == ["granted", "granted", "not_eligible"]
    assert api.called("POST", f"{ADMIN}/groups/msg-1/access")[0] == {"subjectType": "user", "subjectId": "u-bob"}
    again, _ = await call_tool(ctx, "grant_model_service_access", {"service_id": service, "subjects": ["user:bob"]})
    assert again["results"][0]["status"] == "already_granted"
    nobody, is_error = await call_tool(
        ctx, "grant_model_service_access", {"service_id": service, "subjects": ["user:eve"]}
    )
    assert is_error and nobody["error"]["field"] == "subjects"

    revoked, is_error = await call_tool(
        ctx,
        "revoke_model_service_access",
        {"service_id": service, "subjects": ["user:bob", "user:eve"], "token_ids": ["tok-1", "tok-9"]},
    )
    assert not is_error, revoked
    assert [r["status"] for r in revoked["results"]] == ["revoked", "not_granted", "revoked", "not_found"]
    assert [g["subjectName"] for g in lens.grants] == ["ml"]


@pytest.mark.asyncio
async def test_revoke_validates_its_arguments(ctx, lens):
    payload, is_error = await call_tool(ctx, "revoke_model_service_access", {})
    assert is_error and payload["error"]["code"] == "invalid_argument"
    payload, is_error = await call_tool(ctx, "revoke_model_service_access", {"subjects": ["user:bob"]})
    assert is_error and payload["error"]["code"] == "invalid_argument"
    payload, is_error = await call_tool(ctx, "revoke_model_service_access", {"token_ids": ["tok-1"]})
    assert not is_error and payload["results"][0]["status"] == "revoked"


@pytest.mark.asyncio
async def test_test_service_without_token_runs_only_control_plane_checks(monkeypatch, ctx, lens):
    monkeypatch.delenv("LENS_MCP_V2_GATEWAY_TOKEN", raising=False)
    service = (await publish(ctx))["service_id"]
    payload, is_error = await call_tool(ctx, "test_model_service", {"service_id": service})
    assert not is_error and payload["ok"] and all(c["ok"] for c in payload["checks"])
    assert payload["inference"]["status"] == "skipped" and "LENS_MCP_V2_GATEWAY_TOKEN" in payload["inference"]["reason"]


@pytest.mark.asyncio
async def test_test_service_sends_requests_with_token_and_never_returns_it(monkeypatch, ctx, lens):
    monkeypatch.setenv("LENS_MCP_V2_GATEWAY_TOKEN", "lens-mk-secret-value-123")
    sent = []

    async def fake_chat(base_url, token, model, prompt, max_tokens):
        sent.append((base_url, token, model))
        return {"ok": True, "http_status": 200, "latency_ms": 5, "untrusted_text": "hello"}

    monkeypatch.setattr("llm_d_lens.mcp_v2.tools.serving.send_chat", fake_chat)
    service = (await publish(ctx))["service_id"]
    payload, is_error = await call_tool(ctx, "test_model_service", {"service_id": service, "requests": 3})
    assert not is_error and payload["ok"] and payload["inference"]["succeeded"] == 3
    assert sent == [("https://gw/v1", "lens-mk-secret-value-123", "qwen")] * 3
    assert "secret-value" not in str(payload)
    payload, is_error = await call_tool(ctx, "test_model_service", {"service_id": service, "requests": 6})
    assert is_error and payload["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_test_service_reports_failed_checks_and_skips_inference(monkeypatch, ctx, lens):
    monkeypatch.setenv("LENS_MCP_V2_GATEWAY_TOKEN", "lens-mk-secret-value-123")
    service = (await publish(ctx))["service_id"]
    await call_tool(ctx, "disable_model_service", {"service_id": service})
    payload, is_error = await call_tool(ctx, "test_model_service", {"service_id": service})
    assert not is_error and payload["ok"] is False and payload["inference"]["status"] == "skipped"
    assert {"check": "service_enabled", "ok": False} in payload["checks"]


@pytest.mark.asyncio
async def test_model_service_kind_lists_blocks_and_deletes(api, ctx, lens):
    service = (await publish(ctx))["service_id"]
    lens.grants.append({"id": "b0", "subjectType": "user", "subjectId": "u-bob", "subjectName": "bob"})
    listed, _ = await call_tool(ctx, "list_resources", {"kinds": ["model_service"], "include": ["dependents"]})
    (entry,) = listed["resources"]
    assert entry["live_status"] == "active" and {d["kind"] for d in entry["dependents"]} == {
        "deployment",
        "access_grant",
    }

    blocked, _ = await call_tool(ctx, "delete_resources", {"ids": [service]})
    assert blocked["results"][0]["status"] == "blocked" and lens.group is not None
    done, _ = await call_tool(ctx, "delete_resources", {"ids": [service], "cascade": True})
    assert done["deleted"] == [service] and lens.group is None
    assert ctx.ledger.get(service).state == "deleted"


@pytest.mark.asyncio
async def test_list_others_shows_foreign_services_read_only(api, ctx, lens):
    await publish(ctx)
    lens.group_foreign = {"id": "msg-9", "name": "bobs", "clusterId": "c1", "status": "active"}
    api.on("GET", f"{ADMIN}/groups", lambda p, b: {"items": [lens.group, lens.group_foreign]})
    listed, _ = await call_tool(ctx, "list_resources", {"kinds": ["model_service"], "owner": "all"})
    names = {e["name"]: e["ownership"] for e in listed["resources"]}
    assert names == {"qwen": "owned", "bobs": "unknown"}
