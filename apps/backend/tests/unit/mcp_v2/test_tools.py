import pytest

from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.planning import serving_image
from llm_d_lens.mcp_v2.registry import all_tools, call_tool

from .conftest import FakeApi, load_plan, plan_routes

PLAN_ARGS = {"cluster_id": "c1", "guide": "optimized-baseline", "model": "Qwen/Qwen3-0.6B"}


def test_every_tool_declares_honest_annotations():
    tools = {spec.name: spec for spec in all_tools()}
    assert {"plan_deployment", "get_deployment", "get_logs", "list_clusters"} <= {
        n for n, s in tools.items() if s.read_only
    }
    assert tools["delete_deployment"].destructive and tools["delete_deployment"].idempotent
    assert not tools["create_deployment"].read_only and not tools["create_deployment"].destructive
    assert all(not spec.destructive for spec in tools.values() if spec.read_only)


def test_serving_image_reads_modelserver_container():
    plan = load_plan()
    assert serving_image(plan["plannedDeployment"]["content"], plan["deployment"]["readinessDeployments"]) == (
        "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    )


@pytest.mark.asyncio
async def test_plan_infers_accelerator_and_repeats_guide_image_in_render(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "plan_deployment", PLAN_ARGS)
    assert not is_error, payload
    assert payload["accelerator"] == "xpu" and payload["variant"] == "base" and payload["model_server"] == "vllm"
    (render,) = api.called("POST", "/configurations/render")
    assert render["candidate_config"]["runtime"]["image"] == "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    assert render["candidate_config"]["candidate_source"] == {"name": "manual"}
    assert api.called("POST", "/deployments/runs") == []
    assert ctx.ledger.get_plan(payload["plan_id"], ctx.owner) is not None


@pytest.mark.asyncio
async def test_plan_rejects_unsupported_accelerator_with_hint(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "plan_deployment", {**PLAN_ARGS, "accelerator": "gpu"})
    assert is_error
    assert payload["error"]["code"] == "unsupported_accelerator"
    assert payload["error"]["field"] == "accelerator" and "xpu" in payload["error"]["hint"]


@pytest.mark.asyncio
async def test_invalid_arguments_name_the_field(ctx):
    payload, is_error = await call_tool(ctx, "plan_deployment", {**PLAN_ARGS, "replicas": 0})
    assert is_error and payload["error"]["code"] == "invalid_argument" and payload["error"]["field"] == "replicas"


@pytest.mark.asyncio
async def test_plan_with_validation_errors_is_not_creatable(api, ctx):
    plan_routes(api)
    bad = {**load_plan(), "status": "INVALID", "validation": {"status": "invalid", "errors": ["needs 40 accelerators"]}}
    api.on("POST", "/guide-planning/plan", bad)
    payload, is_error = await call_tool(ctx, "plan_deployment", PLAN_ARGS)
    assert is_error and payload["error"]["code"] == "plan_invalid"
    assert "needs 40 accelerators" in payload["error"]["message"]


async def create(api: FakeApi, ctx: ToolContext) -> str:
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1", "status": "queued", "cases": []})
    api.on("GET", "/deployments/runs/run-1", {"id": "run-1", "cases": [{"execution_id": "exec-1"}]})
    plan, _ = await call_tool(ctx, "plan_deployment", PLAN_ARGS)
    created, is_error = await call_tool(ctx, "create_deployment", {"plan_id": plan["plan_id"]})
    assert not is_error, created
    return created["deployment_id"]


@pytest.mark.asyncio
async def test_create_tags_provenance_records_ledger_and_consumes_plan(api, ctx):
    deployment_id = await create(api, ctx)
    (run,) = api.called("POST", "/deployments/runs")
    assert len(run["configurations"]) == 1
    assert run["provenance"]["created_via"] == "lens-mcp-v2" and run["provenance"]["ledger_id"] == deployment_id
    assert run["provenance"]["cluster_server_id"] == "c1"
    row = ctx.ledger.get(deployment_id)
    assert row.owner == "user-a" and row.lens_run_id == "run-1" and row.lens_execution_id == "exec-1"
    reuse, is_error = await call_tool(ctx, "create_deployment", {"plan_id": "plan-unknown"})
    assert is_error and reuse["error"]["code"] == "plan_not_found"


@pytest.mark.asyncio
async def test_failed_create_leaves_no_active_ledger_row(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", LensApiError(503, {"detail": "no capacity"}, "POST", "/deployments/runs"))
    plan, _ = await call_tool(ctx, "plan_deployment", PLAN_ARGS)
    payload, is_error = await call_tool(ctx, "create_deployment", {"plan_id": plan["plan_id"]})
    assert is_error and payload["error"]["upstream_status"] == 503
    assert ctx.ledger.list_resources(owner=ctx.owner, kind="deployment") == []


@pytest.mark.asyncio
async def test_delete_removes_own_deployment_and_is_idempotent(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("DELETE", "/deployments/executions/exec-1", None)
    payload, is_error = await call_tool(ctx, "delete_deployment", {"deployment_id": deployment_id})
    assert not is_error and payload["deleted"] is True
    assert api.called("DELETE", "/deployments/executions/exec-1") == [{"delete_namespace": True}]
    again, is_error = await call_tool(ctx, "delete_deployment", {"deployment_id": deployment_id})
    assert not is_error and again["deleted"] is True
    assert len(api.called("DELETE", "/deployments/executions/exec-1")) == 1


@pytest.mark.asyncio
async def test_delete_refuses_deployments_not_in_the_ledger(api, ctx):
    payload, is_error = await call_tool(ctx, "delete_deployment", {"deployment_id": "exec-of-someone-else"})
    assert is_error and payload["error"]["code"] == "not_owned"
    assert not any(method == "DELETE" for method, _, _ in api.calls)


@pytest.mark.asyncio
async def test_delete_refuses_another_users_ledger_entry(api, ctx):
    deployment_id = await create(api, ctx)
    other = ToolContext(api=FakeApi(owner="user-b"), ledger=ctx.ledger, owner="user-b")
    payload, is_error = await call_tool(other, "delete_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "deployment_not_found"
    by_execution, is_error = await call_tool(other, "delete_deployment", {"deployment_id": "exec-1"})
    assert is_error and by_execution["error"]["code"] == "not_owned"


@pytest.mark.asyncio
async def test_delete_blocked_by_lens_keeps_deployment_owned(api, ctx):
    deployment_id = await create(api, ctx)
    api.on(
        "DELETE",
        "/deployments/executions/exec-1",
        LensApiError(409, {"detail": "in use", "code": "deployment_in_use"}, "DELETE", "/x"),
    )
    payload, is_error = await call_tool(ctx, "delete_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "deployment_in_use"
    assert ctx.ledger.get(deployment_id).state == "active"


@pytest.mark.asyncio
async def test_get_deployment_flags_ownership_and_pending_hint(api, ctx):
    deployment_id = await create(api, ctx)
    api.on(
        "GET",
        "/deployments/executions/exec-1",
        {"execution_id": "exec-1", "status": "deploying", "namespace": "ns", "name": "n"},
    )
    api.on(
        "GET",
        "/deployments/executions/exec-1/pods",
        {"pods": [{"name": "p1", "phase": "Pending", "ready": False, "restarts": 0, "status_reason": "Unschedulable"}]},
    )
    mine, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    assert mine["owned"] is True and mine["ready"] is False and "free accelerators" in mine["hint"]
    theirs, _ = await call_tool(ctx, "get_deployment", {"deployment_id": "exec-1"})
    assert theirs["deployment_id"] == deployment_id
    other = ToolContext(api=api, ledger=ctx.ledger, owner="user-b")
    foreign, _ = await call_tool(other, "get_deployment", {"deployment_id": "exec-1"})
    assert foreign["owned"] is False and foreign["deployment_id"] == "exec-1"


@pytest.mark.asyncio
async def test_list_deployments_defaults_to_mine(api, ctx):
    deployment_id = await create(api, ctx)
    api.on(
        "GET",
        "/deployments/executions",
        {"items": [{"execution_id": "exec-1", "status": "ready"}, {"execution_id": "exec-9", "status": "ready"}]},
    )
    mine, _ = await call_tool(ctx, "list_deployments", {})
    assert [d["deployment_id"] for d in mine["deployments"]] == [deployment_id]
    everyone, _ = await call_tool(ctx, "list_deployments", {"mine_only": False})
    assert {d["deployment_id"]: d["owned"] for d in everyone["deployments"]} == {deployment_id: True, "exec-9": False}


@pytest.mark.asyncio
async def test_get_logs_defaults_to_a_pod_that_is_not_ready(api, ctx):
    deployment_id = await create(api, ctx)
    api.on(
        "GET",
        "/deployments/executions/exec-1/pods",
        {"pods": [{"name": "ok", "ready": True}, {"name": "bad", "ready": False, "restarts": 3}]},
    )
    api.on("GET", "/deployments/executions/exec-1/pods/bad/logs", {"logs": "OOMKilled", "success": True})
    payload, is_error = await call_tool(ctx, "get_logs", {"deployment_id": deployment_id})
    assert (
        not is_error and payload["pod"] == "bad" and payload["logs"] == "OOMKilled" and payload["other_pods"] == ["ok"]
    )
