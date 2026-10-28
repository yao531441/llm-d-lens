"""End-to-end API tests for authentication and route enforcement."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_d_lens.api.main import app
from llm_d_lens.auth.settings import AuthSettings, set_settings_for_testing

STRONG = "Correct-Horse-9!"  # nosemgrep - test fixture
NEW_STRONG = "Another-Strong-7!"  # nosemgrep - test fixture


@pytest.fixture(autouse=True)
def _enable_local_auth(tmp_path_factory, monkeypatch):
    """These tests exercise enforcement, so enable local auth for the module.

    The master-key store is redirected to a temporary file so startup never
    writes to the operator's real ``credentials/`` directory.
    """
    from llm_d_lens.auth import master_key

    key_dir = tmp_path_factory.mktemp("master-key")
    monkeypatch.setattr(master_key, "master_key_path", lambda: key_dir / "master_key.json")
    set_settings_for_testing(
        AuthSettings(
            auth_mode="local",
            session_ttl_seconds=3600,
            session_idle_seconds=3600,
            auto_seed_admin=False,
        )
    )
    yield


def _client() -> TestClient:
    # HTTPS base URL so `Secure` session/CSRF cookies are sent by the client.
    return TestClient(app, base_url="https://testserver")


def _bootstrap_and_login(client: TestClient, username: str = "root", password: str = STRONG) -> None:
    response = client.post("/api/v1/auth/bootstrap", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    login = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200, login.text


def _csrf_headers(client: TestClient) -> dict[str, str]:
    return {"x-prism-csrf": client.cookies.get("prism_csrf")}


def test_health_is_public():
    with _client() as client:
        assert client.get("/api/health").status_code == 200


def test_hidden_migrated_routes_participate_in_startup_validation(monkeypatch):
    from llm_d_lens.auth import routes

    prefixes = (
        "/api/config",
        "/api/mcp",
        "/api/playground",
        "/api/candidate-",
        "/api/deploy-poc",
        "/api/remote-deploy",
        "/api/guide-planning",
    )
    expected = {(entry.method, entry.path) for entry in routes.ROUTE_PERMISSIONS if entry.path.startswith(prefixes)}
    assert len(expected) == 29
    assert expected <= routes.app_route_table(app)
    assert routes.validate_route_registry(app) == []
    monkeypatch.setattr(
        routes,
        "ROUTE_PERMISSIONS",
        tuple(entry for entry in routes.ROUTE_PERMISSIONS if (entry.method, entry.path) != ("POST", "/api/mcp")),
    )
    assert routes.validate_route_registry(app) == ["route not registered: POST /api/mcp"]


def test_session_cookie_max_age_uses_absolute_ttl_not_idle_window():
    """The session cookie must outlive the idle window.

    Otherwise an active user gets logged out the instant the (shorter) idle
    timeout has elapsed since login, even though the server-side
    `idle_expires_at` would have kept sliding forward on every request --
    exactly the "logged out at 30 minutes regardless of activity" bug this
    guards against.
    """
    set_settings_for_testing(
        AuthSettings(
            auth_mode="local",
            session_ttl_seconds=7200,
            session_idle_seconds=1800,
            auto_seed_admin=False,
        )
    )
    with _client() as client:
        _bootstrap_and_login(client)
        login = client.post("/api/v1/auth/login", json={"username": "root", "password": STRONG})
        set_cookie_headers = login.headers.get_list("set-cookie")
        session_cookie = next(header for header in set_cookie_headers if header.startswith("prism_session="))
        assert "Max-Age=7200" in session_cookie
        assert "Max-Age=1800" not in session_cookie


def test_protected_route_requires_authentication():
    with _client() as client:
        response = client.get("/api/v1/users")
        assert response.status_code == 401
        assert response.json()["code"] == "unauthenticated"


def _migrated_routes():
    from llm_d_lens.auth.routes import ROUTE_PERMISSIONS

    prefixes = (
        "/api/configurations/",
        "/api/candidate-search",
        "/api/candidate-support",
        "/api/deploy-poc/",
        "/api/remote-deploy/",
        "/api/guide-planning/",
    )
    routes = [(entry.method, entry.path) for entry in ROUTE_PERMISSIONS if entry.path.startswith(prefixes)]
    assert len(routes) >= 20
    return [*routes, ("POST", "/api/playground/chat"), ("POST", "/api/mcp")]


def test_migrated_business_routes_require_authentication():
    with _client() as client:
        for method, path in _migrated_routes():
            response = client.request(method, path, json={})
            assert response.status_code == 401, (method, path, response.text)
            assert response.json()["code"] == "unauthenticated"


@pytest.mark.parametrize("permission", ["mcp:tool:invoke", "playground:chat:use"])
def test_mcp_entry_permission_does_not_grant_target_tool_permission(monkeypatch, permission):
    import json

    from httpx_sse import EventSource

    from llm_d_lens.auth.contracts import Principal, RoleBinding, ScopeType
    from llm_d_lens.auth.service import default_service

    principal = Principal(
        user_id="limited",
        username="limited",
        bindings=(RoleBinding(role="limited", scope_type=ScopeType.GLOBAL),),
        role_permissions={"limited": frozenset({permission})},
        role_self_scoped={"limited": frozenset()},
        authz_version=0,
    )
    monkeypatch.setattr(type(default_service()), "validate", lambda _self, _token: principal)

    def rpc(client, method, params):
        response = client.post(
            "/api/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            headers={"accept": "application/json, text/event-stream", "authorization": "Bearer fixture"},
        )
        assert response.status_code == 200, response.text
        return json.loads(next(EventSource(response).iter_sse()).data)

    with _client() as client:
        assert client.post("/api/mcp", json={}).status_code == 401
        assert rpc(client, "tools/list", {})["result"]["tools"]
        result = rpc(client, "tools/call", {"name": "list_deploy_poc_configs", "arguments": {}})
        body = json.loads(result["result"]["content"][0]["text"])
        assert "status 403" in body["error"]
        assert body["detail"]["code"] == "forbidden"


def test_migrated_business_mutations_require_csrf():
    with _client() as client:
        _bootstrap_and_login(client)
        for method, path in _migrated_routes():
            if method != "POST":
                continue
            response = client.post(path, json={})
            assert response.status_code == 403, (path, response.text)
            assert response.json()["code"] == "csrf_failed"


def test_authenticated_playground_forwards_session_and_csrf_to_provider_and_tools(monkeypatch):
    import json
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx

    from llm_d_lens.ai_providers import router as provider_router
    from llm_d_lens.playground import chat

    monkeypatch.setattr(
        "llm_d_lens.ai_providers.service._resolve_provider_addresses", AsyncMock(return_value=["8.8.8.8"])
    )
    monkeypatch.setattr(chat, "discover_max_model_len", AsyncMock(return_value=100000))
    upstream = SimpleNamespace(
        post=AsyncMock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {"name": "list_deploy_poc_configs", "arguments": "{}"},
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                ),
                httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "answer"}}]}),
            ]
        )
    )

    @asynccontextmanager
    async def provider_client(_base_url, _timeout):
        yield upstream

    monkeypatch.setattr(provider_router, "public_provider_client", provider_client)
    with _client() as client:
        _bootstrap_and_login(client)
        provider = client.post(
            "/api/v1/ai-providers",
            json={"name": "Chat test", "baseUrl": "https://provider.example.com", "model": "test-model"},
            headers=_csrf_headers(client),
        )
        assert provider.status_code == 201, provider.text
        body = {
            "deployment": {"providerId": provider.json()["id"], "model": "test-model"},
            "messages": [{"role": "user", "content": "List deployment configurations without making changes"}],
            "readOnly": True,
        }
        for headers in ({}, {"x-prism-csrf": "mismatched-token"}):
            rejected = client.post("/api/playground/chat", json=body, headers=headers)
            assert rejected.status_code == 403
            assert rejected.json()["code"] == "csrf_failed"
        upstream.post.assert_not_awaited()

        response = client.post("/api/playground/chat", json=body, headers=_csrf_headers(client))
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "content-encoding" not in response.headers
        events = [
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
            for frame in response.text.strip().split("\n\n")
            if (lines := frame.splitlines())
        ]
        assert not {"error", "permission_denied", "blocked"} & {event for event, _ in events}, events
        result = next(data for event, data in events if event == "tool_result")
        assert result["isError"] is False
        assert json.loads(result["result"]["content"][0]["text"]) == client.get("/api/deploy-poc/config").json()
        assert events[-2:] == [("message", {"content": "answer"}), ("done", {})]
        assert upstream.post.await_count == 2
        assert any(
            message.get("tool_call_id") == "call-1" for message in upstream.post.call_args.kwargs["json"]["messages"]
        )


def test_authenticated_mcp_browser_session_initializes_and_calls_read_tool():
    import json

    from httpx_sse import EventSource

    with _client() as client:
        _bootstrap_and_login(client)

        def rpc(method, params):
            response = client.post(
                "/api/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                headers={**_csrf_headers(client), "accept": "application/json, text/event-stream"},
            )
            assert response.status_code == 200, response.text
            return json.loads(next(EventSource(response).iter_sse()).data)["result"]

        initialized = rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "browser-test", "version": "1"},
            },
        )
        assert initialized["serverInfo"]["name"] == "llm-d-prism"
        listed = rpc("tools/list", {})
        assert "list_deploy_poc_configs" in {tool["name"] for tool in listed["tools"]}
        called = rpc("tools/call", {"name": "list_deploy_poc_configs", "arguments": {}})
        assert not called.get("isError"), called
        assert json.loads(called["content"][0]["text"]) == client.get("/api/deploy-poc/config").json()


def test_migrated_business_routes_deny_unprivileged_users():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/users", json={"username": "migration-user", "password": STRONG}, headers=_csrf_headers(client)
        )
        assert created.status_code == 201
    with _client() as client:
        login = client.post("/api/v1/auth/login", json={"username": "migration-user", "password": STRONG})
        assert login.status_code == 200
        for method, path in _migrated_routes():
            response = client.request(method, path, json={}, headers=_csrf_headers(client))
            assert response.status_code == 403, (method, path, response.text)
            assert response.json()["code"] == "forbidden"


def test_bootstrap_login_session_and_admin_access():
    with _client() as client:
        _bootstrap_and_login(client)
        session = client.get("/api/v1/auth/session")
        assert session.status_code == 200
        assert session.json()["username"] == "root"

        users = client.get("/api/v1/users")
        assert users.status_code == 200

        create = client.post(
            "/api/v1/users",
            json={"username": "alice", "password": STRONG},
            headers=_csrf_headers(client),
        )
        assert create.status_code == 201, create.text
        assert "password_hash" not in create.json()


def test_permission_denied_for_end_user():
    with _client() as client:
        _bootstrap_and_login(client)
        client.post(
            "/api/v1/users",
            json={"username": "bob", "password": STRONG},
            headers=_csrf_headers(client),
        )
        # Give bob the built-in end-user role so he can authenticate scoped.
        roles = client.get("/api/v1/roles").json()
        end_user = next(role for role in roles if role["name"] == "end-user")
        bob = next(user for user in client.get("/api/v1/users").json() if user["username"] == "bob")
        client.post(
            f"/api/v1/users/{bob['id']}/role-bindings",
            json={"roleId": end_user["id"], "scopeType": "global"},
            headers=_csrf_headers(client),
        )

    with _client() as client:
        login = client.post("/api/v1/auth/login", json={"username": "bob", "password": STRONG})
        assert login.status_code == 200
        # end-user has no system:database:read.
        denied = client.get("/api/v1/system/database")
        assert denied.status_code == 403
        # end-user is model-service-only (D15): workload/deployment access is denied.
        assert client.get("/api/v1/deployments/runs").status_code == 403
        # But model service consumption is allowed.
        assert client.get("/api/v1/model-service/models").status_code == 200


def test_csrf_required_for_mutating_requests():
    with _client() as client:
        _bootstrap_and_login(client)
        # Without the CSRF header the mutation is rejected.
        missing = client.post("/api/v1/users", json={"username": "carol", "password": STRONG})
        assert missing.status_code == 403
        assert missing.json()["code"] == "csrf_failed"
        # With the header it succeeds.
        ok = client.post(
            "/api/v1/users",
            json={"username": "carol", "password": STRONG},
            headers=_csrf_headers(client),
        )
        assert ok.status_code == 201


def test_password_change_required_blocks_other_routes():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/users",
            json={"username": "dave", "password": STRONG},
            headers=_csrf_headers(client),
        ).json()
        reset = client.post(
            f"/api/v1/users/{created['id']}/password",
            json={"newPassword": NEW_STRONG},
            headers=_csrf_headers(client),
        )
        assert reset.status_code == 204

    with _client() as client:
        login = client.post("/api/v1/auth/login", json={"username": "dave", "password": NEW_STRONG})
        assert login.status_code == 200
        assert login.json()["principal"]["mustChangePassword"] is True
        blocked = client.get("/api/v1/deployments/runs")
        assert blocked.status_code == 403
        assert blocked.json()["code"] == "password_change_required"


def test_cluster_and_storage_visibility_is_scope_filtered(monkeypatch):
    from llm_d_lens.db.dao.cluster import ClusterDao

    async def _not_ready(_cluster):
        return False

    monkeypatch.setattr("llm_d_lens.cluster.service.cluster_is_ready", _not_ready)
    from llm_d_lens.db.dao.storage_volume import StorageVolumeDao
    from llm_d_lens.storage.contracts import (
        LocalDiskSpec,
        StorageVolume,
        StorageVolumeKind,
        StorageVolumeStatus,
    )

    cluster_a = ClusterDao().create("scope-a", "", "apiVersion: v1\nkind: Config")
    cluster_b = ClusterDao().create("scope-b", "", "apiVersion: v1\nkind: Config")
    for cluster_id, name in ((cluster_a.id, "vol-a"), (cluster_b.id, "vol-b")):
        StorageVolumeDao().create(
            StorageVolume(
                clusterId=cluster_id,
                name=name,
                kind=StorageVolumeKind.LOCAL_DISK,
                status=StorageVolumeStatus.READY,
                capacity="10Gi",
                readOnly=True,
                localDisk=LocalDiskSpec(hostPath="/data/models"),
                pvcName=f"pvc-{name}",
            )
        )

    with _client() as client:
        _bootstrap_and_login(client)
        roles = client.get("/api/v1/roles").json()
        maintainer = next(role for role in roles if role["name"] == "maintainer")
        bob = client.post(
            "/api/v1/users",
            json={"username": "scopebob", "password": STRONG},
            headers=_csrf_headers(client),
        ).json()
        client.post(
            f"/api/v1/users/{bob['id']}/role-bindings",
            json={"roleId": maintainer["id"], "scopeType": "cluster", "scopeClusterId": cluster_a.id},
            headers=_csrf_headers(client),
        )
        # Admin (global) sees both clusters.
        admin_response = client.get("/api/cluster/clusters")
        assert admin_response.status_code == 200, admin_response.text
        admin_ids = {cluster["id"] for cluster in admin_response.json()["items"]}
        assert {cluster_a.id, cluster_b.id} <= admin_ids

    with _client() as client:
        assert client.post("/api/v1/auth/login", json={"username": "scopebob", "password": STRONG}).status_code == 200
        clusters_response = client.get("/api/cluster/clusters")
        assert clusters_response.status_code == 200, clusters_response.text
        bob_ids = {cluster["id"] for cluster in clusters_response.json()["items"]}
        guide_response = client.get("/api/guide-planning/catalog", params={"clusterId": cluster_b.id})
        assert guide_response.status_code == 403, guide_response.text
        assert bob_ids == {cluster_a.id}

        volumes_response = client.get("/api/v1/storage/volumes")
        assert volumes_response.status_code == 200, volumes_response.text
        volume_names = {volume["name"] for volume in volumes_response.json()["items"]}
        assert volume_names == {"vol-a"}


def test_first_run_seeds_admin_and_password_change_clears_credentials(monkeypatch, tmp_path):
    from llm_d_lens.auth import bootstrap

    cred_file = tmp_path / "credentials" / "initial_admin.txt"
    monkeypatch.setattr(bootstrap, "initial_admin_path", lambda: cred_file)
    set_settings_for_testing(
        AuthSettings(auth_mode="local", auto_seed_admin=True, session_ttl_seconds=3600, session_idle_seconds=3600)
    )

    with _client() as client:
        assert cred_file.exists()
        values = {}
        for line in cred_file.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key] = value
        login = client.post(
            "/api/v1/auth/login",
            json={"username": values["username"], "password": values["password"]},
        )
        assert login.status_code == 200, login.text
        assert login.json()["principal"]["mustChangePassword"] is True

        change = client.post(
            "/api/v1/auth/password",
            json={"oldPassword": values["password"], "newPassword": NEW_STRONG},
            headers=_csrf_headers(client),
        )
        assert change.status_code == 204, change.text
        assert not cred_file.exists()
        # The caller's own session survives so they can continue working.
        session = client.get("/api/v1/auth/session")
        assert session.status_code == 200
        assert session.json()["mustChangePassword"] is False

        # Sign out and re-login with the new password works; the old one fails.
        logout = client.post("/api/v1/auth/logout", headers=_csrf_headers(client))
        assert logout.status_code == 204, logout.text
        relogin = client.post(
            "/api/v1/auth/login",
            json={"username": values["username"], "password": NEW_STRONG},
        )
        assert relogin.status_code == 200, relogin.text
        old = client.post(
            "/api/v1/auth/login",
            json={"username": values["username"], "password": values["password"]},
        )
        assert old.status_code == 401


def test_patch_user_csrf_and_unregistered_route_denied():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/users",
            json={"username": "erin", "password": STRONG},
            headers=_csrf_headers(client),
        ).json()
        patched = client.patch(
            f"/api/v1/users/{created['id']}",
            json={"displayName": "Erin"},
            headers=_csrf_headers(client),
        )
        assert patched.status_code == 200
        assert patched.json()["display_name"] == "Erin"


def test_my_sessions_endpoint_returns_list():
    with _client() as client:
        _bootstrap_and_login(client)
        response = client.get("/api/v1/auth/sessions")
        assert response.status_code == 200, response.text
        assert isinstance(response.json(), list)


def test_record_list_endpoints_serialize():
    with _client() as client:
        _bootstrap_and_login(client)
        for path in (
            "/api/v1/auth/sessions",
            "/api/v1/sessions",
            "/api/v1/groups",
            "/api/v1/roles",
            "/api/v1/identity-providers",
            "/api/v1/audit-logs",
        ):
            response = client.get(path)
            assert response.status_code == 200, (path, response.text)
        user_id = client.get("/api/v1/auth/session").json()["userId"]
        for path in (
            f"/api/v1/users/{user_id}/groups",
            f"/api/v1/users/{user_id}/role-bindings",
        ):
            response = client.get(path)
            assert response.status_code == 200, (path, response.text)


def test_groups_list_includes_role_bindings():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/groups",
            json={"name": "ops", "description": ""},
            headers=_csrf_headers(client),
        )
        assert created.status_code == 201, created.text
        group_id = created.json()["id"]
        end_user = next(role for role in client.get("/api/v1/roles").json() if role["name"] == "end-user")
        binding = client.post(
            f"/api/v1/groups/{group_id}/role-bindings",
            json={"roleId": end_user["id"], "scopeType": "global"},
            headers=_csrf_headers(client),
        )
        assert binding.status_code == 201, binding.text

        target = next(item for item in client.get("/api/v1/groups").json() if item["id"] == group_id)
        assert target["roles"] == [
            {
                "name": "end-user",
                "scopeType": "global",
                "scopeClusterId": None,
                "scopeResourceType": None,
                "scopeResourceId": None,
            }
        ]


def test_users_list_includes_roles_inherited_from_groups():
    with _client() as client:
        _bootstrap_and_login(client)
        user = client.post(
            "/api/v1/users",
            json={"username": "carol", "password": STRONG},
            headers=_csrf_headers(client),
        )
        assert user.status_code == 201, user.text
        user_id = user.json()["id"]
        group = client.post(
            "/api/v1/groups",
            json={"name": "ops", "description": ""},
            headers=_csrf_headers(client),
        )
        assert group.status_code == 201, group.text
        group_id = group.json()["id"]
        end_user = next(role for role in client.get("/api/v1/roles").json() if role["name"] == "end-user")
        client.post(
            f"/api/v1/groups/{group_id}/role-bindings",
            json={"roleId": end_user["id"], "scopeType": "global"},
            headers=_csrf_headers(client),
        )
        membership = client.put(
            f"/api/v1/groups/{group_id}/members",
            json={"userIds": [user_id]},
            headers=_csrf_headers(client),
        )
        assert membership.status_code == 204, membership.text

        target = next(item for item in client.get("/api/v1/users").json() if item["id"] == user_id)
        assert target["roles"] == []
        assert target["inheritedRoles"] == [
            {
                "name": "end-user",
                "scopeType": "global",
                "scopeClusterId": None,
                "scopeResourceType": None,
                "scopeResourceId": None,
                "groupId": group_id,
                "groupName": "ops",
            }
        ]


def test_audit_logs_support_pagination_and_filters():
    with _client() as client:
        _bootstrap_and_login(client)

        page = client.get("/api/v1/audit-logs?limit=1&offset=0")
        assert page.status_code == 200, page.text
        body = page.json()
        assert set(body) == {"items", "total", "limit", "offset"}
        assert len(body["items"]) == 1
        assert body["total"] >= 1
        assert body["limit"] == 1 and body["offset"] == 0

        success = client.get("/api/v1/audit-logs?result=success&limit=200").json()
        assert success["total"] >= 1
        assert all(item["result"] == "success" for item in success["items"])

        user_id = client.get("/api/v1/auth/session").json()["userId"]
        by_actor = client.get(f"/api/v1/audit-logs?actor_user_id={user_id}&limit=200").json()
        assert all(item["actor_user_id"] == user_id for item in by_actor["items"])

        # A time range that cannot contain anything returns an empty page.
        assert client.get("/api/v1/audit-logs?since=2999-01-01T00:00:00Z").json()["total"] == 0


def test_protected_flag_marks_the_builtin_admin():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/users",
            json={"username": "admin", "password": STRONG},
            headers=_csrf_headers(client),
        )
        assert created.status_code == 201, created.text
        users = {user["username"]: user for user in client.get("/api/v1/users").json()}
        assert users["admin"]["protected"] is True
        assert users["root"]["protected"] is False


def test_session_list_is_admin_scoped_with_usernames():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/users", json={"username": "alice", "password": STRONG}, headers=_csrf_headers(client)
        )
        assert created.status_code == 201, created.text

        # A non-admin only sees their own sessions (no username enrichment).
        client.post("/api/v1/auth/login", json={"username": "alice", "password": STRONG})
        own = client.get("/api/v1/sessions").json()
        assert own
        assert all(session.get("username") is None for session in own)

        # An admin sees every session, enriched with usernames.
        client.post("/api/v1/auth/login", json={"username": "root", "password": STRONG})
        all_sessions = client.get("/api/v1/sessions").json()
        assert {"root", "alice"} <= {session.get("username") for session in all_sessions}


def test_configuration_artifacts_list_returns_200():
    with _client() as client:
        _bootstrap_and_login(client)
        response = client.get("/api/v1/configurations/artifacts")
        assert response.status_code == 200, response.text


def test_master_key_status_rotate_and_clear():
    with _client() as client:
        _bootstrap_and_login(client)

        status = client.get("/api/v1/system/secret-key")
        assert status.status_code == 200, status.text
        body = status.json()
        assert body["configured"] is True
        assert body["fingerprint"]
        assert body["envLocked"] is False

        rotated = client.post(
            "/api/v1/system/secret-key/rotate",
            json={"newKey": "api-rotated-master-key"},
            headers=_csrf_headers(client),
        )
        assert rotated.status_code == 200, rotated.text
        assert rotated.json()["oldKeyCount"] == 1
        assert rotated.json()["fingerprint"] != body["fingerprint"]

        cleared = client.delete("/api/v1/system/secret-key/old", headers=_csrf_headers(client))
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["oldKeyCount"] == 0


def test_audit_pipeline_records_request_correlation_and_redacts_secrets():
    with _client() as client:
        _bootstrap_and_login(client)
        request_id = "audit-pipeline-test"
        created = client.post(
            "/api/v1/users",
            json={"username": "audited-user", "password": STRONG},
            headers={**_csrf_headers(client), "x-request-id": request_id},
        )
        assert created.status_code == 201, created.text

        page = client.get(f"/api/v1/audit-logs?request_id={request_id}&limit=50").json()
        events = {item["event_type"]: item for item in page["items"]}
        assert "users.create" in events
        event = events["users.create"]
        assert event["action"] == "create"
        assert event["target_type"] == "users"
        assert event["target_name"] == "audited-user"
        assert event["source"] == "browser"
        assert event["request_id"] == request_id
        assert event["detail"]["after"]["password_hash"] == "[REDACTED]"  # noqa: S105

        filtered = client.get("/api/v1/audit-logs?action=create&target_type=users&limit=50").json()
        assert any(item["id"] == event["id"] for item in filtered["items"])


def test_audit_pipeline_tracks_create_update_and_delete_actions():
    with _client() as client:
        _bootstrap_and_login(client)
        created = client.post(
            "/api/v1/groups",
            json={"name": "audit-actions", "description": ""},
            headers=_csrf_headers(client),
        )
        assert created.status_code == 201, created.text
        group_id = created.json()["id"]
        updated = client.patch(
            f"/api/v1/groups/{group_id}",
            json={"description": "changed"},
            headers=_csrf_headers(client),
        )
        assert updated.status_code == 200, updated.text
        deleted = client.delete(f"/api/v1/groups/{group_id}", headers=_csrf_headers(client))
        assert deleted.status_code == 204, deleted.text

        page = client.get(f"/api/v1/audit-logs?target_type=groups&target_id={group_id}&limit=50").json()
        assert {"create", "update", "delete"} <= {item["action"] for item in page["items"]}


def test_login_bookkeeping_audit_uses_the_signed_in_user_as_actor():
    with _client() as client:
        _bootstrap_and_login(client)

        page = client.get("/api/v1/audit-logs?event_type=users.update&limit=50").json()

        assert page["items"]
        assert all(item["actor_username"] == "root" for item in page["items"])


def test_audit_update_events_are_suppressed_for_high_frequency_workloads():
    from llm_d_lens.db.base import audit_tracks_write

    for table in ("simulation_tasks", "evaluate_runs", "evaluate_workflows", "evaluate_workflow_cases"):
        assert audit_tracks_write(table, "create")
        assert not audit_tracks_write(table, "update")
        assert audit_tracks_write(table, "delete")

    assert audit_tracks_write("clusters", "update")
