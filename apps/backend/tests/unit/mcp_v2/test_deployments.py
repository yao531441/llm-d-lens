import asyncio

import pytest

from llm_d_lens.mcp_v2 import deploy_test, deployment_refs
from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.planning import serving_image
from llm_d_lens.mcp_v2.registry import all_tools, call_tool
from llm_d_lens.mcp_v2.tools import observe

from .conftest import FakeApi, load_plan, plan_routes

SPEC = {"cluster_id": "c1", "guide": "optimized-baseline", "model": "Qwen/Qwen3-0.6B"}
EXEC = "exec-1"


def test_deployment_tools_declare_honest_annotations():
    tools = {spec.name: spec for spec in all_tools()}
    assert "delete_deployment" not in tools and "list_deployments" not in tools
    reads = ("plan_deployment", "get_deployment", "diagnose_deployment", "get_logs")
    assert all(tools[name].read_only and not tools[name].destructive for name in reads)
    assert tools["stop_deployment"].destructive and tools["stop_deployment"].idempotent
    assert not tools["create_deployment"].read_only and not tools["create_deployment"].destructive
    assert not tools["test_deployment"].read_only and not tools["test_deployment"].idempotent
    assert tools["update_deployment"].idempotent and tools["restart_deployment"].idempotent
    assert all(tools[name].profile == "core" for name in (*reads, "create_deployment", "test_deployment"))


def test_serving_image_reads_modelserver_container():
    plan = load_plan()
    assert serving_image(plan["plannedDeployment"]["content"], plan["deployment"]["readinessDeployments"]) == (
        "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    )


def pod(name, role, *, ready=True, phase="Running", restarts=0, reason=None, container="vllm"):
    return {
        "name": name,
        "labels": {"llm-d.ai/role": role} if role else {"app": "harness"},
        "phase": phase,
        "ready": ready,
        "restarts": restarts,
        "status_reason": reason,
        "containers": [{"name": container, "ready": ready, "state": "running", "restart_count": restarts}],
    }


def lens_state(api: FakeApi, *, status="ready", pods=(), case_status="ready", execution_id=EXEC, **execution):
    api.on("GET", "/deployments/runs/run-1", {"id": "run-1", "cases": [lens_case(case_status, execution_id)]})
    api.on(
        "GET",
        f"/deployments/executions/{execution_id}",
        {"execution_id": execution_id, "status": status, "namespace": "ns", "cluster_id": "c1", **execution},
    )
    api.on("GET", f"/deployments/executions/{execution_id}/pods", {"pods": list(pods)})


def lens_case(status, execution_id=EXEC, *, attempt=1):
    return {"id": f"case-{attempt}", "attempt": attempt, "status": status, "execution_id": execution_id}


async def create(api: FakeApi, ctx: ToolContext, spec=None, **extra) -> str:
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1", "status": "queued", "cases": []})
    lens_state(api)
    created, is_error = await call_tool(ctx, "create_deployment", {"spec": spec or SPEC, **extra})
    assert not is_error, created
    return created["deployment_id"]


# plan ----------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_infers_accelerator_and_repeats_guide_image_in_render(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "plan_deployment", {"spec": SPEC})
    assert not is_error, payload
    assert payload["status"] in ("valid", "valid_with_warnings")
    assert payload["resolved_spec"]["accelerator"] == "xpu" and payload["resolved_spec"]["variant"] == "base"
    (render,) = api.called("POST", "/configurations/render")
    assert render["candidate_config"]["runtime"]["image"] == "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    assert render["candidate_config"]["candidate_source"] == {"name": "manual"}
    assert api.called("POST", "/deployments/runs") == []
    assert ctx.ledger.get_plan(payload["plan_id"], ctx.owner) is not None


@pytest.mark.asyncio
async def test_plan_reports_missing_capacity_as_warning(api, ctx):
    plan_routes(api)
    spec = {**SPEC, "topology": {"serving": {"replicas": 30}}}
    payload, is_error = await call_tool(ctx, "plan_deployment", {"spec": spec})
    assert not is_error, payload
    assert any(w["code"] == "insufficient_capacity" for w in payload["validation"]["warnings"])
    assert payload["resources"]["fits_now"] is False


@pytest.mark.asyncio
async def test_plan_rejects_unsupported_accelerator_as_invalid_status(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "plan_deployment", {"spec": {**SPEC, "accelerator": "gpu"}})
    assert not is_error and payload["status"] == "invalid"
    (problem,) = payload["validation"]["errors"]
    assert problem["code"] == "unsupported_accelerator" and "xpu" in problem["hint"]


@pytest.mark.asyncio
async def test_plan_with_lens_validation_errors_is_invalid_and_not_creatable(api, ctx):
    plan_routes(api)
    bad = {**load_plan(), "status": "INVALID", "validation": {"status": "invalid", "errors": ["needs 40 accelerators"]}}
    api.on("POST", "/guide-planning/plan", bad)
    payload, _ = await call_tool(ctx, "plan_deployment", {"spec": SPEC})
    assert payload["status"] == "invalid" and "needs 40 accelerators" in payload["validation"]["errors"][0]["message"]
    created, is_error = await call_tool(ctx, "create_deployment", {"spec": SPEC})
    assert is_error and created["error"]["code"] == "plan_invalid"
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_plan_rejects_pd_topology_on_other_guides(api, ctx):
    plan_routes(api)
    spec = {**SPEC, "topology": {"prefill": {"replicas": 1}}}
    payload, _ = await call_tool(ctx, "plan_deployment", {"spec": spec})
    assert payload["status"] == "invalid" and payload["validation"]["errors"][0]["code"] == "unsupported_combination"


@pytest.mark.asyncio
async def test_recommend_is_labelled_heuristic_and_checks_capacity(api, ctx):
    plan_routes(api)
    overview = api.routes[("GET", "/cluster/overview")]
    overview["kubernetes"]["hardware"].update(vramBytes=32 * 24 * 1024**3, gpuCount=32)
    api.on(
        "POST",
        "/capacity/estimate",
        lambda params, body: {"is_deployable": body["tensor_parallel_size"] >= 2, "rejection_reasons": ["too big"]},
    )
    payload, is_error = await call_tool(
        ctx, "plan_deployment", {"mode": "recommend", "spec": {"cluster_id": "c1", "model": "Qwen/Qwen3-0.6B"}}
    )
    assert not is_error, payload
    assert payload["heuristic"] is True
    best = payload["candidates"][0]
    assert best["spec"]["guide"] == "optimized-baseline"
    assert best["spec"]["topology"]["serving"]["tensor_parallel_size"] == 2 and best["fits_now"] is True


@pytest.mark.asyncio
async def test_invalid_arguments_name_the_field(ctx):
    payload, is_error = await call_tool(
        ctx, "plan_deployment", {"spec": {**SPEC, "topology": {"serving": {"replicas": 0}}}}
    )
    assert is_error and payload["error"]["code"] == "invalid_argument"


# create --------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sends_one_configuration_with_provenance_and_records_ledger(api, ctx):
    deployment_id = await create(api, ctx, ttl_hours=4)
    (body,) = api.called("POST", "/deployments/runs")
    assert len(body["configurations"]) == 1
    assert body["provenance"]["ledger_id"] == deployment_id and body["provenance"]["created_via"] == "lens-mcp-v2"
    assert body["provenance"]["cluster_server_id"] == "c1"
    row = ctx.ledger.get(deployment_id)
    assert row.state == "active" and row.lens_run_id == "run-1" and row.owner == ctx.owner
    assert row.meta["spec"]["model"] == "Qwen/Qwen3-0.6B"
    assert 3.9 * 3600 < row.meta["expires_at"] - __import__("time").time() <= 4 * 3600


@pytest.mark.asyncio
async def test_create_returns_id_and_next_hint_and_consumes_plan(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1"})
    plan, _ = await call_tool(ctx, "plan_deployment", {"spec": SPEC})
    created, is_error = await call_tool(ctx, "create_deployment", {"plan_id": plan["plan_id"]})
    assert not is_error, created
    assert created["deployment_id"].startswith("dep-") and created["status"] == "pending"
    assert "wait_seconds=25" in created["next"]
    assert ctx.ledger.get_plan(plan["plan_id"], ctx.owner) is None
    again, is_error = await call_tool(ctx, "create_deployment", {"plan_id": plan["plan_id"]})
    assert is_error and again["error"]["code"] == "plan_expired"


@pytest.mark.asyncio
async def test_create_requires_exactly_one_source_and_bounded_ttl(api, ctx):
    plan_routes(api)
    for args in ({}, {"spec": SPEC, "plan_id": "plan-x"}):
        payload, is_error = await call_tool(ctx, "create_deployment", args)
        assert is_error and payload["error"]["code"] == "invalid_argument"
    for ttl in (0, 73):
        payload, is_error = await call_tool(ctx, "create_deployment", {"spec": SPEC, "ttl_hours": ttl})
        assert is_error and payload["error"]["code"] == "invalid_argument"
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_create_rejected_by_lens_with_4xx_leaves_no_live_ledger_row(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", LensApiError(422, {"detail": "bad manifest"}, "POST", "/deployments/runs"))
    payload, is_error = await call_tool(ctx, "create_deployment", {"spec": SPEC})
    assert is_error and payload["error"]["code"] != "upstream_unavailable"
    assert ctx.ledger.list_resources(owner=ctx.owner, kind="deployment") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [LensApiError(502, {"detail": "gateway"}, "POST", "/deployments/runs"), TimeoutError()]
)
async def test_create_with_uncertain_outcome_keeps_the_row_and_reconciles_by_provenance(api, ctx, failure):
    plan_routes(api)
    api.on("POST", "/deployments/runs", failure)
    payload, is_error = await call_tool(ctx, "create_deployment", {"spec": SPEC})
    assert is_error and payload["error"]["code"] == "upstream_unavailable"
    deployment_id = payload["error"]["deployment_id"]
    assert "get_deployment" in payload["error"]["hint"]
    assert ctx.ledger.get(deployment_id).state == "creating"

    # Lens never recorded a run: still waiting inside the grace period.
    api.on("GET", "/deployments/runs", [])
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    assert got["status"] == "pending" and ctx.ledger.get(deployment_id).state == "creating"

    # The run exists after all: it is matched through provenance.ledger_id and bound.
    api.on("GET", "/deployments/runs", [{"id": "run-9", "provenance": {"ledger_id": deployment_id}}])
    api.on("GET", "/deployments/runs/run-9", {"id": "run-9", "cases": [lens_case("deploying", None)]})
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    row = ctx.ledger.get(deployment_id)
    assert row.state == "active" and row.lens_run_id == "run-9" and got["status"] == "starting"


@pytest.mark.asyncio
async def test_unconfirmed_create_that_never_appears_in_lens_is_dropped_after_the_grace_period(api, ctx, monkeypatch):
    plan_routes(api)
    api.on("POST", "/deployments/runs", TimeoutError())
    payload, _ = await call_tool(ctx, "create_deployment", {"spec": SPEC})
    deployment_id = payload["error"]["deployment_id"]
    api.on("GET", "/deployments/runs", [])
    monkeypatch.setattr(deployment_refs, "CREATE_GRACE_SECONDS", -1)
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    assert got["status"] == "deleted" and ctx.ledger.get(deployment_id).state == "deleted"


async def unconfirmed_create(api, ctx) -> str:
    plan_routes(api)
    api.on("POST", "/deployments/runs", TimeoutError())
    payload, _ = await call_tool(ctx, "create_deployment", {"spec": SPEC})
    return payload["error"]["deployment_id"]


@pytest.mark.asyncio
async def test_delete_of_an_unconfirmed_create_waits_then_reconciles_the_run(api, ctx, monkeypatch):
    deployment_id = await unconfirmed_create(api, ctx)
    api.on("GET", "/deployments/runs", [])
    payload, _ = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    (result,) = payload["results"]
    assert result["status"] == "blocked" and result["error"] == "in_use"
    assert ctx.ledger.get(deployment_id).state == "creating"

    # Lens did create the run after all: the delete must remove it instead of just marking the row deleted.
    api.on("GET", "/deployments/runs", [{"id": "run-9", "provenance": {"ledger_id": deployment_id}}])
    api.on("GET", "/deployments/runs/run-9", {"id": "run-9", "cases": [lens_case("deploying", None)]})
    api.on("DELETE", "/deployments/runs/run-9", {})
    payload, _ = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    assert payload["results"][0]["status"] == "deleted"
    assert len(api.called("DELETE", "/deployments/runs/run-9")) == 1
    assert ctx.ledger.get(deployment_id).state == "deleted"


@pytest.mark.asyncio
async def test_delete_of_an_unconfirmed_create_after_the_grace_period_just_drops_the_row(api, ctx, monkeypatch):
    deployment_id = await unconfirmed_create(api, ctx)
    api.on("GET", "/deployments/runs", [])
    monkeypatch.setattr(deployment_refs, "CREATE_GRACE_SECONDS", -1)
    payload, _ = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    assert payload["results"][0]["status"] == "deleted"
    assert api.called("DELETE", "/deployments/runs/run-9") == []
    assert ctx.ledger.get(deployment_id).state == "deleted"


@pytest.mark.asyncio
async def test_experiment_launched_rows_are_left_to_the_experiment_engine(api, ctx, monkeypatch):
    row = ctx.ledger.add_resource(
        kind="deployment",
        owner=ctx.owner,
        cluster_id="c1",
        name="v1",
        root_id="exp-1",
        state="creating",
        meta={"spec": SPEC, "experiment_id": "exp-1"},
    )
    api.on("GET", "/deployments/runs", [])
    monkeypatch.setattr(deployment_refs, "CREATE_GRACE_SECONDS", -1)
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": row.id})
    assert got["status"] == "pending" and ctx.ledger.get(row.id).state == "creating"
    assert api.called("GET", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_cancelled_create_leaves_the_row_creating(api, ctx):
    plan_routes(api)

    def cancelled(params, body):
        raise asyncio.CancelledError

    api.on("POST", "/deployments/runs", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await call_tool(ctx, "create_deployment", {"spec": SPEC})
    (row,) = ctx.ledger.list_resources(owner=ctx.owner, kind="deployment")
    assert row.state == "creating"


@pytest.mark.asyncio
async def test_create_refuses_duplicate_name_on_cluster(api, ctx):
    await create(api, ctx, {**SPEC, "name": "demo"})
    payload, is_error = await call_tool(ctx, "create_deployment", {"spec": {**SPEC, "name": "demo"}})
    assert is_error and payload["error"]["code"] == "already_exists"


@pytest.mark.asyncio
async def test_create_from_deployment_patches_stored_spec_and_names_a_fresh_deployment(api, ctx):
    first = await create(api, ctx, {**SPEC, "name": "demo"})
    api.on("POST", "/deployments/runs", {"id": "run-2"})
    created, is_error = await call_tool(
        ctx,
        "create_deployment",
        {"from_deployment_id": first, "spec_patch": {"topology": {"serving": {"replicas": 2}}}},
    )
    assert not is_error, created
    row = ctx.ledger.get(created["deployment_id"])
    assert row.meta["spec"]["topology"]["serving"]["replicas"] == 2 and row.name != "demo"
    assert ctx.ledger.get(first).meta["spec"]["topology"]["serving"]["replicas"] != 2


@pytest.mark.asyncio
async def test_create_from_someone_elses_deployment_is_not_found(api, ctx):
    foreign = ctx.ledger.add_resource(kind="deployment", owner="user-b", cluster_id="c1", meta={"spec": SPEC})
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "create_deployment", {"from_deployment_id": foreign.id})
    assert is_error and payload["error"]["code"] in ("not_found", "deployment_not_found")


# get_deployment ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_deployment_reports_roles_endpoint_and_spec(api, ctx):
    deployment_id = await create(api, ctx, {**SPEC, "topology": {"serving": {"replicas": 2}}})
    lens_state(
        api,
        status="deploying",
        pods=[
            pod("p1", "decode"),
            pod("p2", "decode", ready=False),
            pod("w", None),
            {**pod("epp", None), "labels": {"llm-d-router-gateway": "x"}},
        ],
    )
    got, is_error = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    assert not is_error, got
    assert got["status"] == "starting" and got["ready"] is False and got["owned"] is True
    assert got["roles"]["decode"] == {"ready": 1, "total": 2, "desired": 2}
    assert "router" in got["roles"] and len(got["roles"]) == 2  # the harness pod is not part of the deployment
    assert got["spec"]["model"] == "Qwen/Qwen3-0.6B" and got["observed_at"] and got["expires_at"]


@pytest.mark.asyncio
async def test_get_deployment_by_raw_execution_id_is_visible_but_not_owned(api, ctx):
    lens_state(api, pods=[pod("p1", "decode")], model="m", guide="optimized-baseline")
    got, is_error = await call_tool(ctx, "get_deployment", {"deployment_id": EXEC})
    assert not is_error and got["owned"] is False and got["status"] == "ready"
    assert got["spec"]["model"] == "m"
    api.routes.pop(("GET", f"/deployments/executions/{EXEC}"))
    missing, is_error = await call_tool(ctx, "get_deployment", {"deployment_id": EXEC})
    assert is_error and missing["error"]["code"] == "deployment_not_found"


@pytest.mark.asyncio
async def test_get_deployment_of_other_users_ledger_id_is_not_found(api, ctx):
    foreign = ctx.ledger.add_resource(kind="deployment", owner="user-b", cluster_id="c1", lens_execution_id=EXEC)
    payload, is_error = await call_tool(ctx, "get_deployment", {"deployment_id": foreign.id})
    assert is_error and payload["error"]["code"] == "deployment_not_found"


@pytest.mark.asyncio
async def test_get_deployment_long_poll_returns_on_status_change(api, ctx, monkeypatch):
    monkeypatch.setattr(observe, "POLL_SECONDS", 0.01)
    deployment_id = await create(api, ctx)
    lens_state(api, status="queued")
    reads = {"n": 0}

    def execution(params, body):
        reads["n"] += 1
        return {"execution_id": EXEC, "status": "queued" if reads["n"] < 3 else "ready", "namespace": "ns"}

    api.on("GET", f"/deployments/executions/{EXEC}", execution)
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id, "wait_seconds": 5})
    assert got["status"] == "ready" and reads["n"] == 3


@pytest.mark.asyncio
async def test_get_deployment_restart_check_compares_needed_and_free_accelerators(api, ctx):
    deployment_id = await create(
        api, ctx, {**SPEC, "topology": {"serving": {"replicas": 2, "tensor_parallel_size": 2}}}
    )
    lens_state(api, status="rolled_back", case_status="stopped")
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id, "include": ["restart_check"]})
    check = got["restart_check"]
    assert got["status"] == "stopped" and check["status_allows_restart"] is True
    assert check["accelerators_needed"] == 4 and check["accelerators_free"] == 26 and check["fits_now"] is True


# ownership -----------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["stop_deployment", "restart_deployment", "update_deployment", "test_deployment"])
async def test_mutations_refuse_deployments_not_owned(api, ctx, tool):
    lens_state(api)
    ctx.ledger.add_resource(kind="deployment", owner="user-b", cluster_id="c1", lens_execution_id="exec-b")
    base = {"name": "x"} if tool == "update_deployment" else {}
    payload, is_error = await call_tool(ctx, tool, {"deployment_id": EXEC, **base})
    assert is_error and payload["error"]["code"] == "owner_unknown"
    payload, is_error = await call_tool(ctx, tool, {"deployment_id": "exec-b", **base})
    assert is_error and payload["error"]["code"] == "not_owner"


# update / stop / restart ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_name_description_ttl_labels_and_monitoring(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("PATCH", f"/deployments/executions/{EXEC}", {})
    api.on("POST", f"/monitoring/deployments/{EXEC}/enable", {"status": "enabled"})
    updated, is_error = await call_tool(
        ctx,
        "update_deployment",
        {
            "deployment_id": deployment_id,
            "name": "renamed",
            "description": "d",
            "ttl_hours": 48,
            "labels": {"team": "a"},
            "monitoring": True,
        },
    )
    assert not is_error, updated
    assert api.called("PATCH", f"/deployments/executions/{EXEC}") == [{"display_name": "renamed", "description": "d"}]
    assert api.called("POST", f"/monitoring/deployments/{EXEC}/enable") == [{"cluster_id": "c1"}]
    row = ctx.ledger.get(deployment_id)
    assert row.name == "renamed" and row.meta["labels"] == {"team": "a"}
    assert 47 * 3600 < row.meta["expires_at"] - __import__("time").time() <= 48 * 3600


@pytest.mark.asyncio
async def test_update_rejects_empty_changes_and_out_of_range_ttl(api, ctx):
    deployment_id = await create(api, ctx)
    for args in ({}, {"ttl_hours": 0}, {"ttl_hours": 100}):
        payload, is_error = await call_tool(ctx, "update_deployment", {"deployment_id": deployment_id, **args})
        assert is_error and payload["error"]["code"] == "invalid_argument"


STOP = "/deployments/runs/run-1/cases/case-1/stop"


@pytest.mark.asyncio
async def test_stop_posts_to_the_latest_case_and_is_idempotent(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("POST", STOP, {})
    stopped, is_error = await call_tool(ctx, "stop_deployment", {"deployment_id": deployment_id})
    assert not is_error and stopped["status"] == "stopped"
    # The Lens route is POST; any other method on this path is a bug.
    assert [method for method, path, _ in api.calls if path == STOP] == ["POST"]
    lens_state(api, status="rolled_back", case_status="stopped")
    again, _ = await call_tool(ctx, "stop_deployment", {"deployment_id": deployment_id})
    assert again["already_stopped"] is True
    assert len(api.called("POST", STOP)) == 1


@pytest.mark.asyncio
async def test_stop_conflict_from_lens_is_actionable(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("POST", STOP, LensApiError(409, {"detail": "no active cluster session"}, "POST", STOP))
    payload, is_error = await call_tool(ctx, "stop_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "conflict_state" and payload["error"]["hint"]


@pytest.mark.asyncio
async def test_restart_requires_stopped_or_failed(api, ctx):
    deployment_id = await create(api, ctx)
    payload, is_error = await call_tool(ctx, "restart_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "conflict_state"
    assert "stop_deployment" in payload["error"]["hint"]


@pytest.mark.asyncio
async def test_restart_never_calls_the_broken_lens_restart_and_points_to_a_rebuild(api, ctx):
    # Lens restart_case cleans the namespace, marks the case CLEANED, then retry_case rejects it.
    deployment_id = await create(api, ctx)
    lens_state(api, status="rolled_back", case_status="stopped")
    api.on("POST", "/deployments/runs/run-1/cases/case-1/restart", {})
    payload, is_error = await call_tool(ctx, "restart_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "conflict_state"
    assert f"from_deployment_id='{deployment_id}'" in payload["error"]["hint"]
    assert api.called("POST", "/deployments/runs/run-1/cases/case-1/restart") == []


@pytest.mark.asyncio
async def test_ledger_rebinds_to_the_new_execution_after_a_lens_retry(api, ctx):
    deployment_id = await create(api, ctx)
    ctx.ledger.bind_execution(deployment_id, EXEC)
    ctx.ledger.update_meta(deployment_id, {"restarting_from": EXEC})
    api.on(
        "GET",
        "/deployments/runs/run-1",
        {"cases": [lens_case("stopped", EXEC), lens_case("queued", EXEC, attempt=2)]},
    )
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    assert got["status"] == "pending" and ctx.ledger.get(deployment_id).lens_execution_id == EXEC

    lens_state(api, status="deploying", execution_id="exec-2", case_status="deploying")
    api.on(
        "GET",
        "/deployments/runs/run-1",
        {"cases": [lens_case("stopped", EXEC), lens_case("deploying", "exec-2", attempt=2)]},
    )
    got, _ = await call_tool(ctx, "get_deployment", {"deployment_id": deployment_id})
    assert got["deployment_id"] == deployment_id and got["status"] == "starting"
    assert ctx.ledger.get(deployment_id).lens_execution_id == "exec-2"


def borrowing_experiment(ctx, deployment_id, *, phase="running"):
    return ctx.ledger.add_resource(
        kind="experiment",
        owner=ctx.owner,
        cluster_id="c1",
        name="exp",
        meta={"phase": phase, "variants": {"a": {"deploy": "borrowed", "dep_id": deployment_id}}},
    )


@pytest.mark.asyncio
async def test_running_experiment_borrowing_a_deployment_blocks_stop_and_delete(api, ctx):
    deployment_id = await create(api, ctx)
    ctx.ledger.bind_execution(deployment_id, EXEC)
    api.on("POST", STOP, {})
    api.on("DELETE", f"/deployments/executions/{EXEC}", {})
    experiment = borrowing_experiment(ctx, deployment_id)

    payload, is_error = await call_tool(ctx, "stop_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "in_use" and "cancel_experiment" in payload["error"]["hint"]
    assert api.called("POST", STOP) == []

    for cascade in (False, True):
        payload, _ = await call_tool(ctx, "delete_resources", {"ids": [deployment_id], "cascade": cascade})
        (result,) = payload["results"]
        assert result["status"] == "blocked" and result["error"] == "in_use"
    assert api.called("DELETE", f"/deployments/executions/{EXEC}") == []

    listed, _ = await call_tool(ctx, "list_resources", {"kinds": ["deployment"], "include": ["dependents"]})
    assert [d["id"] for d in listed["resources"][0]["dependents"]] == [experiment.id]

    ctx.ledger.update_meta(experiment.id, {"phase": "settled"})
    payload, _ = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    assert payload["results"][0]["status"] == "deleted"


# kind: delete / list_others ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_resources_removes_namespace_and_marks_ledger_deleted(api, ctx):
    deployment_id = await create(api, ctx)
    ctx.ledger.bind_execution(deployment_id, EXEC)
    api.on("DELETE", f"/deployments/executions/{EXEC}", {})
    payload, is_error = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    assert not is_error, payload
    assert api.called("DELETE", f"/deployments/executions/{EXEC}") == [{"delete_namespace": True}]
    assert ctx.ledger.get(deployment_id).state == "deleted"


@pytest.mark.asyncio
async def test_delete_falls_back_to_the_run_and_treats_404_as_done(api, ctx):
    deployment_id = await create(api, ctx)
    ctx.ledger.bind_execution(deployment_id, EXEC)  # no DELETE route for the execution: Lens answers 404
    api.on("DELETE", "/deployments/runs/run-1", {})
    payload, is_error = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    assert not is_error, payload
    assert len(api.called("DELETE", "/deployments/runs/run-1")) == 1
    assert ctx.ledger.get(deployment_id).state == "deleted"


@pytest.mark.asyncio
async def test_delete_in_use_deployment_is_blocked_not_deleted(api, ctx):
    deployment_id = await create(api, ctx)
    ctx.ledger.bind_execution(deployment_id, EXEC)
    api.on(
        "DELETE",
        f"/deployments/executions/{EXEC}",
        LensApiError(409, {"code": "deployment_in_use", "detail": "used by evaluation"}, "DELETE", "x"),
    )
    payload, is_error = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    assert payload["results"][0]["status"] == "blocked" and ctx.ledger.get(deployment_id).state == "active"


@pytest.mark.asyncio
async def test_delete_409_without_code_is_a_blocked_conflict_with_a_hint(api, ctx):
    deployment_id = await create(api, ctx)
    ctx.ledger.bind_execution(deployment_id, EXEC)
    api.on(
        "DELETE",
        f"/deployments/executions/{EXEC}",
        LensApiError(409, {"detail": "deployment is still running"}, "DELETE", "x"),
    )
    payload, _ = await call_tool(ctx, "delete_resources", {"ids": [deployment_id]})
    (result,) = payload["results"]
    assert result["status"] == "blocked" and result["error"] == "conflict_state"
    assert "stop_deployment" in result["message"]
    assert ctx.ledger.get(deployment_id).state == "active"


@pytest.mark.asyncio
async def test_list_resources_marks_other_executions_shared_or_unknown(api, ctx):
    mine = await create(api, ctx)
    ctx.ledger.bind_execution(mine, EXEC)
    ctx.ledger.add_resource(kind="deployment", owner="user-b", cluster_id="c1", lens_execution_id="exec-b")
    items = [
        {"execution_id": execution_id, "status": status, "display_name": execution_id, "cluster_id": "c1"}
        for execution_id, status in ((EXEC, "ready"), ("exec-b", "ready"), ("exec-z", "ready"), ("exec-old", "cleaned"))
    ]
    api.on("GET", "/deployments/executions", {"items": items})
    payload, is_error = await call_tool(
        ctx, "list_resources", {"kinds": ["deployment"], "owner": "all", "cluster_id": "c1"}
    )
    assert not is_error, payload
    ownership = {row["id"]: row["ownership"] for row in payload["resources"]}
    assert ownership == {mine: "owned", "exec-b": "shared", "exec-z": "unknown"}


# diagnose ------------------------------------------------------------------------------------------------------


def logs(api, pod_name, text, *, previous=False):
    api.on(
        "GET",
        f"/deployments/executions/{EXEC}/pods/{pod_name}/logs",
        lambda params, body: {"success": True, "logs": text, "container": params["container"]},
    )


@pytest.mark.asyncio
async def test_diagnose_pod_state_findings_without_cluster_text_outside_untrusted(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(
        api,
        status="deploying",
        case_status="deploying",
        pods=[
            pod("a", "decode", ready=False, reason="ImagePullBackOff"),
            pod("b", "decode", ready=False, reason="OOMKilled", restarts=1),
            pod("c", "decode", ready=False, reason="CrashLoopBackOff", restarts=5),
            pod("d", "decode", ready=False, phase="Pending", reason="Pending"),
        ],
    )
    for name in "abc":
        logs(api, name, "boot")
    api.on(
        "GET",
        "/deployments/runs/run-1/cases/case-1/logs",
        {"entries": [{"source": "events", "message": "0/4 nodes are available: Insufficient gpu.intel.com/xe."}]},
    )
    result, is_error = await call_tool(ctx, "diagnose_deployment", {"deployment_id": deployment_id})
    assert not is_error, result
    codes = {f["code"] for f in result["findings"]}
    assert {"image_pull_error", "oom_killed", "crash_loop", "unschedulable"} <= codes
    assert result["healthy"] is False
    assert all("Insufficient" not in str(f) for f in result["findings"])
    assert any("Insufficient" in line for entry in result["untrusted_text"] for line in entry["lines"])


@pytest.mark.asyncio
async def test_diagnose_finds_model_download_failure_in_logs_and_capacity_shortage(api, ctx):
    deployment_id = await create(api, ctx, {**SPEC, "topology": {"serving": {"replicas": 30}}})
    lens_state(api, status="deploying", pods=[pod("a", "decode", ready=False)])
    logs(api, "a", "huggingface_hub.errors.GatedRepoError: 401 Client Error. Access to model is restricted")
    result, _ = await call_tool(ctx, "diagnose_deployment", {"deployment_id": deployment_id})
    codes = {f["code"] for f in result["findings"]}
    assert "model_access_denied" in codes
    assert "insufficient_capacity" not in codes  # all pods exist and are scheduled
    lens_state(api, status="queued", pods=[pod("a", "decode", ready=False, phase="Pending", reason="Pending")])
    result, _ = await call_tool(ctx, "diagnose_deployment", {"deployment_id": deployment_id})
    assert "insufficient_capacity" in {f["code"] for f in result["findings"]}


@pytest.mark.asyncio
async def test_diagnose_healthy_deployment_has_no_findings(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("a", "decode")])
    result, _ = await call_tool(ctx, "diagnose_deployment", {"deployment_id": deployment_id})
    assert result["healthy"] is True and result["findings"] == []


# get_logs ------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_logs_defaults_to_unhealthy_pod_and_caps_text(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("good", "decode"), pod("bad", "decode", ready=False, restarts=2)])
    logs(api, "bad", "x" * 30000)
    result, is_error = await call_tool(ctx, "get_logs", {"deployment_id": deployment_id})
    assert not is_error, result
    assert result["pod"] == "bad" and result["container"] == "vllm" and result["truncated"] is True
    assert len(result["untrusted_text"]) == 20000
    assert result["other_pods"] == [{"name": "good", "role": "decode", "ready": True}]


@pytest.mark.asyncio
async def test_get_logs_selects_by_role_and_rejects_unknown_pod_or_container(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("pre", "prefill"), pod("dec", "decode")])
    logs(api, "pre", "prefill output")
    result, _ = await call_tool(ctx, "get_logs", {"deployment_id": deployment_id, "role": "prefill", "previous": True})
    assert result["pod"] == "pre" and result["untrusted_text"] == "prefill output"
    assert api.called("GET", f"/deployments/executions/{EXEC}/pods/pre/logs")[-1]["previous"] is True
    for args in ({"pod": "nope"}, {"role": "router"}, {"container": "nope"}):
        payload, is_error = await call_tool(ctx, "get_logs", {"deployment_id": deployment_id, **args})
        assert is_error and payload["error"]["code"] in ("not_found", "invalid_argument")


# test_deployment -----------------------------------------------------------------------------------------------


class FakeEndpoint:
    def __init__(self, cached=0):
        self.requests = []
        self.cached = cached

    async def __call__(self, method, url, payload):
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "Qwen/Qwen3-0.6B"}]}
        self.requests.append(payload)
        usage = {"prompt_tokens_details": {"cached_tokens": self.cached}} if self.cached is not None else {}
        return 200, {"choices": [{"message": {"content": "hello"}}], "usage": usage}


def serve_logs(api, per_pod_lines):
    """Access-log text per pod; each call returns what the fake endpoint 'has served' so far."""
    for name, counter in per_pod_lines.items():
        api.on(
            "GET",
            f"/deployments/executions/{EXEC}/pods/{name}/logs",
            lambda params, body, counter=counter: {
                "success": True,
                "logs": "\n".join('INFO: "POST /v1/chat/completions HTTP/1.1" 200' for _ in range(counter())),
            },
        )


def ticking(before, after):
    state = {"calls": 0}

    def count():
        state["calls"] += 1
        return before if state["calls"] == 1 else after

    return count


@pytest.mark.asyncio
async def test_test_deployment_chat_and_routing_spread_pass(api, ctx, monkeypatch):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("d1", "decode"), pod("d2", "decode")])
    api.on("POST", f"/deployments/executions/{EXEC}/endpoint", {"endpoint": "http://127.0.0.1:9"})
    endpoint = FakeEndpoint()
    monkeypatch.setattr(deploy_test, "http_call", endpoint)
    serve_logs(api, {"d1": ticking(0, 5), "d2": ticking(0, 5)})
    result, is_error = await call_tool(
        ctx, "test_deployment", {"deployment_id": deployment_id, "checks": ["chat", "routing_spread"], "requests": 10}
    )
    assert not is_error, result
    assert result["verdict"] == "pass" and result["evidence_sufficient"] is True
    assert result["requests_sent"] == 10 == len(endpoint.requests)
    assert result["untrusted_text"][0]["lines"] == ["hello"]


@pytest.mark.asyncio
async def test_test_deployment_routing_spread_fails_when_one_pod_gets_everything(api, ctx, monkeypatch):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("d1", "decode"), pod("d2", "decode")])
    api.on("POST", f"/deployments/executions/{EXEC}/endpoint", {"endpoint": "http://127.0.0.1:9"})
    monkeypatch.setattr(deploy_test, "http_call", FakeEndpoint())
    serve_logs(api, {"d1": ticking(0, 12), "d2": ticking(0, 0)})
    result, _ = await call_tool(
        ctx, "test_deployment", {"deployment_id": deployment_id, "checks": ["routing_spread"], "requests": 12}
    )
    assert result["verdict"] == "fail"


@pytest.mark.asyncio
async def test_test_deployment_prefix_cache_needs_cached_token_counts(api, ctx, monkeypatch):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("d1", "decode")])
    api.on("POST", f"/deployments/executions/{EXEC}/endpoint", {"endpoint": "http://127.0.0.1:9"})
    args = {"deployment_id": deployment_id, "checks": ["prefix_cache"], "requests": 10}
    monkeypatch.setattr(deploy_test, "http_call", FakeEndpoint(cached=None))
    result, _ = await call_tool(ctx, "test_deployment", args)
    assert result["verdict"] == "inconclusive" and result["evidence_sufficient"] is False
    monkeypatch.setattr(deploy_test, "http_call", FakeEndpoint(cached=64))
    result, _ = await call_tool(ctx, "test_deployment", args)
    assert result["verdict"] == "pass" and result["requests_sent"] == 10
    short, is_error = await call_tool(ctx, "test_deployment", {**args, "requests": 3})
    assert is_error and short["error"]["field"] == "requests"


@pytest.mark.asyncio
async def test_test_deployment_unreachable_endpoint_is_inconclusive(api, ctx, monkeypatch):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("d1", "decode")])
    api.on("POST", f"/deployments/executions/{EXEC}/endpoint", {"endpoint": "http://127.0.0.1:9"})

    async def refuse(method, url, payload):
        raise deploy_test.EndpointUnreachableError("ConnectError")

    monkeypatch.setattr(deploy_test, "http_call", refuse)
    result, is_error = await call_tool(ctx, "test_deployment", {"deployment_id": deployment_id})
    assert not is_error and result["verdict"] == "inconclusive" and result["evidence_sufficient"] is False
    assert result["requests_sent"] == 0


@pytest.mark.asyncio
async def test_test_deployment_requires_ready_deployment_and_at_most_20_requests(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, status="deploying", pods=[pod("d1", "decode", ready=False)])
    payload, is_error = await call_tool(ctx, "test_deployment", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "conflict_state"
    payload, is_error = await call_tool(ctx, "test_deployment", {"deployment_id": deployment_id, "requests": 21})
    assert is_error and payload["error"]["code"] == "invalid_argument"
