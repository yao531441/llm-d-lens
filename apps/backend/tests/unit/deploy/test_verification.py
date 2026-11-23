"""Verification budgets, evidence limits and durable Web/MCP lifecycle."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from llm_d_lens.deploy import verification, verification_checks
from llm_d_lens.deploy.contracts import (
    ConfigurationArtifact,
    DeployableConfiguration,
    DeploymentArtifact,
    DeploymentCase,
    DeploymentCaseStatus,
    DeploymentCreateRequest,
    DeploymentEndpoint,
    DeploymentExecution,
    DeploymentRun,
    DeploymentRunStatus,
    DeploymentStatus,
    VersionedPayload,
)
from llm_d_lens.deploy.run_store import JsonDeploymentRunStore
from llm_d_lens.deploy.usage import DeploymentInUseError, register_usage_probe, require_unused
from llm_d_lens.deploy.verification_checks import VerificationRequest

router = importlib.import_module("llm_d_lens.deploy.router")
executions = importlib.import_module("llm_d_lens.deploy.executions")


@pytest.fixture
def store(monkeypatch):
    store = JsonDeploymentRunStore()
    artifact = ConfigurationArtifact(
        schema_version="v1",
        kind="helm-values",
        provider_ref="optimized-baseline",
        checksum="hash",
        content='{"model":{"name":"model-1"}}',
    )
    case = DeploymentCase(
        id="case-1",
        run_id="run-1",
        ordinal=0,
        source_configuration_ordinal=0,
        provider_ref="optimized-baseline",
        component="modelservice",
        status=DeploymentCaseStatus.READY,
        execution_id="exec-1",
        create_request=DeploymentCreateRequest(configuration_artifacts=[artifact]),
    )
    store.create_run(
        DeploymentRun(
            id="run-1",
            status=DeploymentRunStatus.SUCCEEDED,
            cases=[case],
            source_configurations=[
                DeployableConfiguration(
                    type="baseline", format="helm", provider_ref="optimized-baseline", checksum="hash"
                )
            ],
        )
    )
    store.save_execution(
        DeploymentExecution(
            execution_id="exec-1",
            request_id="request-1",
            status=DeploymentStatus.READY,
            artifact=DeploymentArtifact(
                artifact_hash="hash",
                configuration_artifact_ids=[artifact.artifact_id],
                rendered_payload=VersionedPayload(schema_version="v1"),
            ),
            configuration_artifacts=[artifact],
            endpoint=DeploymentEndpoint(url="http://service.ns.svc:8000"),
            namespace="ns",
            diagnostics=VersionedPayload(schema_version="v1", value={"runtime": "original"}),
        )
    )
    monkeypatch.setattr(verification, "_store", store)
    monkeypatch.setattr(executions, "_store", store)
    monkeypatch.setattr(router, "_store", store)
    monkeypatch.setattr(
        verification,
        "ensure_execution_endpoint",
        AsyncMock(return_value=SimpleNamespace(endpoint="http://127.0.0.1:9")),
    )
    return store


def completion(*, cached=None):
    return 200, {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


@pytest.mark.asyncio
async def test_budget_and_evidence_limits_are_enforced(monkeypatch):
    calls = []

    async def send(url, payload):
        calls.append(payload)
        return completion(cached=0)

    monkeypatch.setattr(verification_checks, "http_call", send)
    result = await verification_checks.run_checks(
        "http://localhost:9",
        "model",
        VerificationRequest(
            checks=["chat", "prefix_cache"],
            requests=10,
        ),
    )
    assert len(calls) == result["requests_sent"] == 10
    assert result["requests_completed"] == 10 and result["output_tokens"] == 20
    assert all(call["max_tokens"] == 16 for call in calls)
    assert len({call["messages"][0]["content"] for call in calls[-6:]}) == 1
    assert result["verdict"] == "inconclusive" and not result["evidence_sufficient"]
    assert {item["verdict"] for item in result["checks"][1:]} == {"inconclusive"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cached, expected", [(None, "inconclusive"), (0, "inconclusive"), (64, "pass")])
async def test_prefix_cache_only_passes_with_positive_evidence(monkeypatch, cached, expected):
    monkeypatch.setattr(verification_checks, "http_call", AsyncMock(return_value=completion(cached=cached)))
    result = await verification_checks.run_checks(
        "http://localhost:9", "model", VerificationRequest(checks=["prefix_cache"], requests=10)
    )
    assert result["verdict"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected", [((500, {}), "fail"), ((200, {"choices": "invalid"}), "fail"), (None, "inconclusive")]
)
async def test_chat_failures_do_not_claim_success(monkeypatch, response, expected):
    send = AsyncMock(return_value=response) if response else AsyncMock(side_effect=httpx.ConnectError("offline"))
    monkeypatch.setattr(verification_checks, "http_call", send)
    result = await verification_checks.run_checks("http://localhost:9", "model", VerificationRequest(requests=1))
    assert result["verdict"] == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"requests": 0},
        {"requests": 21},
        {"requests": True},
        {"checks": []},
        {"checks": ["chat", "chat"]},
        {"checks": ["pd_transfer"]},
        {"checks": ["routing_spread"]},
        {"checks": ["rank_distribution"]},
        {"checks": ["prefix_cache"], "requests": 9},
    ],
)
def test_invalid_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        VerificationRequest(**payload)


@pytest.mark.asyncio
async def test_observed_chat_failure_is_not_masked_by_transport_unknown(monkeypatch):
    monkeypatch.setattr(
        verification_checks,
        "http_call",
        AsyncMock(side_effect=[(500, {}), httpx.ConnectError("offline")]),
    )
    result = await verification_checks.run_checks("http://localhost:9", "model", VerificationRequest(requests=2))
    assert result["verdict"] == "fail" and result["evidence_sufficient"]


@pytest.mark.asyncio
async def test_native_task_persists_result_and_survives_runtime_refresh(store, monkeypatch):
    monkeypatch.setattr(verification_checks, "http_call", AsyncMock(return_value=completion()))
    stale = store.get_execution("exec-1")
    record = await router.create_verification("exec-1", VerificationRequest(requests=2))
    await verification._tasks[record["id"]]
    got = await router.get_verification("exec-1", record["id"])
    assert got["status"] == "succeeded" and got["verdict"] == "pass"
    assert got["requests_sent"] == got["requests_completed"] == 2
    assert got["max_output_tokens"] == 32
    stale.status = DeploymentStatus.CLEANED
    stale.diagnostics = VersionedPayload(schema_version="v2", value={"runtime": "refreshed"})
    store.save_execution(stale)
    persisted = store.get_execution("exec-1")
    assert persisted.status == DeploymentStatus.CLEANED
    assert persisted.diagnostics.value["runtime"] == "refreshed"
    assert (await router.list_verifications("exec-1"))["items"] == [got]
    assert verification_checks.http_call.await_count == 2  # reads never replay requests


@pytest.mark.asyncio
async def test_pending_task_blocks_teardown_and_duplicate_submission(store, monkeypatch):
    gate = asyncio.Event()

    async def wait(*_args):
        await gate.wait()
        return completion()

    monkeypatch.setattr(verification_checks, "http_call", wait)
    record = verification.start_verification("exec-1", VerificationRequest(requests=1))
    task = verification._tasks[record["id"]]
    register_usage_probe("verification", verification.active_usage)
    with pytest.raises(DeploymentInUseError):
        require_unused("exec-1")
    with pytest.raises(ValueError, match="already active"):
        verification.start_verification("exec-1", VerificationRequest(requests=1))
    gate.set()
    await task
    require_unused("exec-1")


@pytest.mark.asyncio
async def test_task_fails_at_the_deadline_keeping_the_requests_already_sent(store, monkeypatch):
    calls = 0

    async def slow(*_args):
        nonlocal calls
        calls += 1
        if calls > 1:
            await asyncio.sleep(60)
        return completion()

    monkeypatch.setattr(verification_checks, "http_call", slow)
    monkeypatch.setattr(verification, "TIMEOUT_SECONDS", 0.05)
    record = verification.start_verification("exec-1", VerificationRequest(requests=3))
    await verification._tasks[record["id"]]
    got = verification.get_verification("exec-1", record["id"])
    assert got["status"] == "failed" and "within" in got["error"]
    assert got["requests_sent"] == 1 and got["requests_succeeded"] == 1
    assert verification.active_usage("exec-1") is None


def test_interrupted_records_fail_on_startup_and_history_is_bounded(store):
    store.update_execution_verification("exec-1", {"id": "old", "status": "running"}, create=True)
    verification.fail_interrupted_verifications()
    assert verification.get_verification("exec-1", "old")["status"] == "failed"
    for index in range(25):
        store.update_execution_verification("exec-1", {"id": str(index), "status": "succeeded"}, create=True)
    records = verification.list_verifications("exec-1")
    assert len(records) == 20 and records[-1]["id"] == "5"


@pytest.mark.asyncio
async def test_routes_enforce_scope_before_creating_or_reading_tasks(store, monkeypatch):
    accesses = []

    def deny(principal, context, permission):
        accesses.append((context.execution_id, permission))
        raise HTTPException(403, "permission denied")

    monkeypatch.setattr(router, "_require_execution_permission", deny)
    for call in (
        router.create_verification("exec-1", VerificationRequest()),
        router.list_verifications("exec-1"),
        router.get_verification("exec-1", "missing"),
    ):
        with pytest.raises(HTTPException) as error:
            await call
        assert error.value.status_code == 403
    assert [permission for _, permission in accesses] == [
        "deployment:execution:connect",
        "deployment:run:read",
        "deployment:run:read",
    ]
    assert verification.list_verifications("exec-1") == []


@pytest.mark.asyncio
async def test_mcp_starts_native_task_and_web_reads_identical_result(store, monkeypatch):
    from llm_d_lens.mcp.context import ToolContext
    from llm_d_lens.mcp.lens_api import LensApiError
    from llm_d_lens.mcp.registry import call_tool

    class Api:
        async def request(self, method, path, *, body=None, **_kwargs):
            if path == "/deployments/runs/exec-1":
                raise LensApiError(404, {"detail": "not found"}, method, path)
            if path == "/deployments/executions/exec-1":
                return {"execution_id": "exec-1", "run_id": "run-1", "case_id": "case-1", "ownership": "self"}
            if method == "POST":
                return await router.create_verification("exec-1", VerificationRequest(**body))
            return await router.get_verification("exec-1", path.rsplit("/", 1)[-1])

    monkeypatch.setattr(verification_checks, "http_call", AsyncMock(return_value=completion()))
    ctx = ToolContext(api=Api(), owner="tester")
    started, error = await call_tool(
        ctx, "deployment_test", {"deployment_id": "exec-1", "requests": 2, "wait_seconds": 0}
    )
    assert not error, started
    await verification._tasks[started["verification_id"]]
    web = await router.get_verification("exec-1", started["verification_id"])
    mcp, error = await call_tool(ctx, "deployment_test", {"deployment_id": "exec-1", "verification_id": web["id"]})
    assert not error, mcp
    assert all(mcp[key] == value for key, value in web.items())
    assert verification_checks.http_call.await_count == 2
