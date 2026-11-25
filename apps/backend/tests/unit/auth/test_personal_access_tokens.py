"""Personal credentials share live Lens authorization, without session privileges."""

from contextlib import closing
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from llm_d_lens.api.main import app
from llm_d_lens.auth.records import utcnow
from llm_d_lens.auth.security import hash_session_token
from llm_d_lens.auth.service import default_service
from llm_d_lens.auth.settings import AuthSettings, set_settings_for_testing

PASSWORD = "Correct-Horse-9!"  # noqa: S105 - synthetic test credential


@pytest.fixture
def browser(tmp_path, monkeypatch):
    from llm_d_lens.auth import master_key

    monkeypatch.setattr(master_key, "master_key_path", lambda: tmp_path / "master_key.json")
    set_settings_for_testing(AuthSettings(auth_mode="local", auto_seed_admin=False, internal_auth_secret="test-secret"))  # noqa: S106
    with TestClient(app, base_url="https://testserver") as client:
        assert client.post("/api/v1/auth/bootstrap", json={"username": "root", "password": PASSWORD}).status_code == 200
        assert client.post("/api/v1/auth/login", json={"username": "root", "password": PASSWORD}).status_code == 200
        client.headers["x-prism-csrf"] = client.cookies.get("prism_csrf")
        yield client


def issue(browser):
    response = browser.post("/api/v1/auth/access-tokens", json={"name": "MCP", "expiresInDays": 7})
    assert response.status_code == 201, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def test_hash_only_stable_identity_and_owner_revocation(browser):
    issued = issue(browser)
    service = default_service()
    stored = service.personal_token_dao.get_by_hash(hash_session_token(issued["token"]))
    assert stored.id == issued["id"]
    assert stored.token_hash != issued["token"]
    listed = browser.get("/api/v1/auth/access-tokens").json()
    assert listed[0]["id"] == stored.id
    assert "token_hash" not in issued and "token_hash" not in listed[0] and "token" not in listed[0]
    other = service.create_user(username="other", password=PASSWORD)
    assert not service.personal_token_dao.revoke(stored.id, other.id)
    with closing(
        TestClient(app, base_url="https://testserver", headers={"Authorization": f"Bearer {issued['token']}"})
    ) as api:
        assert api.get("/api/v1/auth/session").json()["userId"] == stored.user_id
        assert service.personal_token_dao.get_by_hash(stored.token_hash).last_used_at is not None
        assert browser.delete(f"/api/v1/auth/access-tokens/{stored.id}").status_code == 204
        assert api.get("/api/v1/auth/session").status_code == 401
    assert browser.get("/api/v1/auth/session").status_code == 200


def _signed(path, *, method="GET", source=None):
    import time

    from llm_d_lens.auth.security import sign_internal

    user = default_service().user_dao.get_by_username("root")
    timestamp = int(time.time())
    headers = {
        "x-prism-principal-id": user.id,
        "x-prism-internal-ts": str(timestamp),
        "x-prism-internal-sig": sign_internal(
            "test-secret", timestamp=timestamp, method=method, path=path, principal_id=user.id
        ),
    }
    return {**headers, "x-prism-source": source} if source else headers


def test_signed_internal_calls_work_but_cannot_manage_credentials(browser):
    with closing(TestClient(app, base_url="https://testserver")) as api:
        path = "/api/v1/auth/access-tokens"
        assert api.get("/api/v1/users", headers=_signed("/api/v1/users")).status_code == 200
        assert api.get(path, headers=_signed(path)).status_code == 403


def test_an_assistant_call_cannot_manage_credentials_even_with_a_login_session(browser):
    path = "/api/v1/auth/access-tokens"
    assert browser.get(path).status_code == 200
    assert browser.get(path, headers={"x-prism-source": "assistant"}).status_code == 403


def test_routes_that_manage_credentials_are_session_only():
    from llm_d_lens.auth.routes import ROUTE_PERMISSIONS

    session_only = {(entry.method, entry.path) for entry in ROUTE_PERMISSIONS if entry.session_only}
    for entry in ROUTE_PERMISSIONS:
        permissions = entry.permission if isinstance(entry.permission, tuple) else (entry.permission or "",)
        manages_secret = any(":token:" in p or ":hf-token:" in p or ":secret:" in p for p in permissions)
        # A new route that handles tokens or secrets must be marked, or listed here on purpose.
        assert not manages_secret or (entry.method, entry.path) in session_only, (entry.method, entry.path)
    assert session_only == {
        ("POST", "/api/v1/auth/password"),
        ("GET", "/api/v1/auth/sessions"),
        ("GET", "/api/v1/auth/access-tokens"),
        ("POST", "/api/v1/auth/access-tokens"),
        ("DELETE", "/api/v1/auth/access-tokens/{token_id}"),
        ("GET", "/api/v1/sessions"),
        ("DELETE", "/api/v1/sessions/{session_id}"),
        ("POST", "/api/v1/users"),
        ("POST", "/api/v1/users/{user_id}/password"),
        ("GET", "/api/v1/identity-providers"),
        ("POST", "/api/v1/identity-providers"),
        ("GET", "/api/v1/identity-providers/{provider_id}"),
        ("PATCH", "/api/v1/identity-providers/{provider_id}"),
        ("DELETE", "/api/v1/identity-providers/{provider_id}"),
        ("POST", "/api/v1/identity-providers/{provider_id}/test"),
        ("POST", "/api/v1/identity-providers/{provider_id}/sync"),
        ("POST", "/api/cluster/clusters/{cluster_id}/hf-token-secrets"),
        ("PUT", "/api/cluster/clusters/{cluster_id}/hf-token-secret"),
        ("POST", "/api/v1/ai-providers"),
        ("DELETE", "/api/v1/ai-providers/{provider_id}"),
        ("PUT", "/api/v1/ai-providers/{provider_id}"),
        ("GET", "/api/v1/system/secret-key"),
        ("POST", "/api/v1/system/secret-key/rotate"),
        ("DELETE", "/api/v1/system/secret-key/old"),
        ("GET", "/api/v1/model-service/tokens"),
        ("POST", "/api/v1/model-service/tokens"),
        ("POST", "/api/v1/model-service/tokens/regenerate"),
        ("DELETE", "/api/v1/model-service/tokens/{token_id}"),
    }


def test_cookie_impersonation_csrf_and_invalid_input(browser):
    issued = issue(browser)
    with closing(TestClient(app, base_url="https://testserver")) as api:
        api.cookies.set("prism_session", issued["token"])
        assert api.get("/api/v1/auth/session").status_code == 401
    assert (
        browser.post("/api/v1/auth/access-tokens", json={"name": "bad"}, headers={"x-prism-csrf": "bad"}).status_code
        == 403
    )
    for days in (0, 366, True, "7"):
        assert (
            browser.post("/api/v1/auth/access-tokens", json={"name": "bad", "expiresInDays": days}).status_code == 422
        )
    assert browser.post("/api/v1/auth/access-tokens", json={"name": "   "}).status_code == 422


def test_live_roles_and_session_independence(browser):
    service = default_service()
    user = service.create_user(username="operator", password=PASSWORD)
    role = service.require_role("admin")
    binding = service.add_user_binding(user.id, role.id)
    record, raw = service.create_personal_token(service.principal_for(user), "client", 1)
    service.revoke_user_sessions(user.id)
    assert service.validate(raw).is_global_admin()
    service.remove_user_binding(user.id, binding.id)
    assert not service.validate(raw).is_global_admin()
    assert service.personal_token_dao.get_by_hash(hash_session_token(raw)).id == record.id


def test_token_metadata_and_revocation_are_owner_only(browser):
    issued = issue(browser)
    service = default_service()
    service.create_user(username="other", password=PASSWORD)
    with closing(TestClient(app, base_url="https://testserver")) as other:
        assert other.post("/api/v1/auth/login", json={"username": "other", "password": PASSWORD}).status_code == 200
        other.headers["x-prism-csrf"] = other.cookies.get("prism_csrf")
        assert other.get("/api/v1/auth/access-tokens").json() == []
        assert other.delete(f"/api/v1/auth/access-tokens/{issued['id']}").status_code == 404
    assert service.validate(issued["token"])


@pytest.mark.parametrize("action", ["disable", "password", "reset", "delete", "expire"])
def test_account_and_expiry_lifecycle(browser, action, monkeypatch):
    service = default_service()
    user = service.create_user(username="operator", password=PASSWORD)
    _, raw = service.create_personal_token(service.principal_for(user), "client", 1)
    assert service.validate(raw)
    if action == "disable":
        service.update_user(user.id, status="disabled")
        service.update_user(user.id, status="active")
    elif action == "password":
        service.change_password(user.id, PASSWORD, "Another-Strong-7!")
    elif action == "reset":
        service.reset_password(user.id, "Another-Strong-7!")
    elif action == "delete":
        service.delete_user(user.id)
    else:
        monkeypatch.setattr("llm_d_lens.auth.service.utcnow", lambda: utcnow() + timedelta(days=2))
    assert service.validate(raw) is None


def test_token_migration_upgrade_downgrade_and_model_parity():
    import importlib

    import sqlalchemy as sa
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from llm_d_lens.db.models.personal_access_token import PersonalAccessTokenRow

    migration = importlib.import_module("llm_d_lens.db.migrations.versions.a6c8e1f3b502_add_personal_access_tokens")
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table("users", metadata, sa.Column("id", sa.String(36), primary_key=True))
    metadata.create_all(engine)
    PersonalAccessTokenRow.__table__.to_metadata(metadata)
    try:
        with engine.begin() as connection:
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                migration.upgrade()
                assert compare_metadata(context, metadata) == []
                migration.downgrade()
                assert "personal_access_tokens" not in sa.inspect(connection).get_table_names()
                migration.upgrade()
                assert compare_metadata(context, metadata) == []
    finally:
        engine.dispose()
