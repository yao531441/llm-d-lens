import pytest

from llm_d_lens.mcp import deployment_refs
from llm_d_lens.mcp.context import ToolContext
from llm_d_lens.mcp.errors import ToolError
from llm_d_lens.mcp.lens_api import LensApiError
from llm_d_lens.mcp.planning import serving_image
from llm_d_lens.mcp.registry import all_tools, call_tool
from llm_d_lens.mcp.tools import observe

from .conftest import FakeApi, load_plan, plan_routes

FIELDS = {"guide": "optimized-baseline", "model": "Qwen/Qwen3-0.6B"}
ARGS = {"cluster_id": "c1", "configuration": FIELDS}
EXEC = "exec-1"
STOP = "/deployments/runs/run-1/cases/case-1/stop"


def test_deployment_tools_declare_honest_annotations():
    tools = {spec.name: spec for spec in all_tools()}
    assert not {"deployment_update", "deployment_restart", "delete_deployment", "list_deployments"} & tools.keys()
    reads = ("deployment_plan", "deployment_get", "deployment_diagnose", "deployment_logs")
    assert all(tools[name].read_only and not tools[name].destructive for name in reads)
    assert tools["deployment_stop"].destructive and tools["deployment_stop"].idempotent
    assert not tools["deployment_create"].read_only and not tools["deployment_create"].destructive
    assert not tools["deployment_test"].read_only and not tools["deployment_test"].idempotent
    assert all(tools[name].profile == "core" for name in (*reads, "deployment_create", "deployment_test"))


def test_serving_image_reads_modelserver_container():
    plan = load_plan()
    assert serving_image(plan["plannedDeployment"]["content"], plan["deployment"]["readinessDeployments"]) == (
        "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    )


def test_serving_image_reads_disaggregated_set_roles():
    manifest = """
apiVersion: disaggregatedset.x-k8s.io/v1
kind: DisaggregatedSet
metadata:
  name: pd-disagg-xpu
spec:
  roles:
    - name: prefill
      spec:
        leaderWorkerTemplate:
          workerTemplate:
            spec:
              containers: [{name: modelserver, image: "example/prefill:1"}]
    - name: decode
      spec:
        leaderWorkerTemplate:
          workerTemplate:
            spec:
              containers: [{name: sidecar, image: "example/sidecar:1"}, {name: modelserver, image: "example/decode:1"}]
"""
    assert serving_image(manifest, ["pd-disagg-xpu"]) == "example/decode:1"


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


def lens_case(status, execution_id=EXEC, *, attempt=1):
    return {"id": f"case-{attempt}", "attempt": attempt, "status": status, "execution_id": execution_id}


def lens_state(api: FakeApi, *, status="ready", pods=(), case_status="ready", execution_id=EXEC, **execution):
    """Lens state of run-1: one case bound to one execution the caller owns, unless overridden."""
    api.on("GET", "/deployments/runs/run-1", {"id": "run-1", "cases": [lens_case(case_status, execution_id)]})
    api.on(
        "GET",
        f"/deployments/executions/{execution_id}",
        {
            "execution_id": execution_id,
            "run_id": "run-1",
            "display_name": "demo",
            "status": status,
            "namespace": "ns",
            "cluster_id": "c1",
            "ownership": "self",
            "capabilities": {"update": True, "stop": True, "delete": True},
            **execution,
        },
    )
    api.on("GET", f"/deployments/executions/{execution_id}/pods", {"pods": list(pods)})


async def create(api: FakeApi, ctx: ToolContext, **extra) -> str:
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1", "status": "queued", "cases": []})
    lens_state(api)
    created, is_error = await call_tool(ctx, "deployment_create", {**ARGS, **extra})
    assert not is_error, created
    return created["deployment_id"]


# plan ----------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_infers_accelerator_and_repeats_guide_image_in_render(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "deployment_plan", ARGS)
    assert not is_error, payload
    assert payload["status"] in ("valid", "valid_with_warnings")
    assert payload["resolved_configuration"]["accelerator"] == "xpu"
    assert payload["resolved_configuration"]["guideVariant"] == "base"
    (render,) = api.called("POST", "/configurations/render")
    assert render["candidate_config"]["runtime"]["image"] == "ghcr.io/llm-d/llm-d-xpu:v1.0.0"
    assert render["candidate_config"]["candidate_source"] == {"name": "manual"}
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_plan_passes_runtime_image_override_to_guide_and_render(api, ctx):
    plan_routes(api)
    image = "registry.example/model-server:custom"

    def planned(params, body):
        assert body["runtimeImage"] == image
        plan = load_plan()
        plan["plannedDeployment"]["content"] = plan["plannedDeployment"]["content"].replace(
            "ghcr.io/llm-d/llm-d-xpu:v1.0.0", image
        )
        return plan

    api.on("POST", "/guide-planning/plan", planned)
    payload, is_error = await call_tool(
        ctx, "deployment_plan", {**ARGS, "configuration": {**FIELDS, "runtimeImage": image}}
    )
    assert not is_error and payload["status"] in ("valid", "valid_with_warnings"), payload
    (render,) = api.called("POST", "/configurations/render")
    assert render["candidate_config"]["runtime"]["image"] == image
    assert serving_image(render["render"]["rendered_manifest"], ["optimized-baseline-xpu-vllm-decode"]) == image


@pytest.mark.asyncio
async def test_plan_reports_missing_capacity_as_warning(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "deployment_plan", {**ARGS, "configuration": {**FIELDS, "replicas": 30}})
    assert not is_error, payload
    assert any(w["code"] == "insufficient_capacity" for w in payload["validation"]["warnings"])
    assert payload["resources"]["fits_now"] is False


@pytest.mark.asyncio
async def test_plan_rejects_unsupported_accelerator_as_invalid_status(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(
        ctx, "deployment_plan", {**ARGS, "configuration": {**FIELDS, "accelerator": "gpu"}}
    )
    assert not is_error and payload["status"] == "invalid"
    (problem,) = payload["validation"]["errors"]
    assert problem["code"] == "unsupported_accelerator" and "xpu" in problem["hint"]


@pytest.mark.asyncio
async def test_plan_with_lens_validation_errors_is_invalid_and_not_creatable(api, ctx):
    plan_routes(api)
    bad = {**load_plan(), "status": "INVALID", "validation": {"status": "invalid", "errors": ["needs 40 accelerators"]}}
    api.on("POST", "/guide-planning/plan", bad)
    payload, _ = await call_tool(ctx, "deployment_plan", ARGS)
    assert payload["status"] == "invalid" and "needs 40 accelerators" in payload["validation"]["errors"][0]["message"]
    created, is_error = await call_tool(ctx, "deployment_create", ARGS)
    assert is_error and created["error"]["code"] == "plan_invalid"
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_plan_rejects_pd_fields_on_other_guides(api, ctx):
    plan_routes(api)
    payload, _ = await call_tool(ctx, "deployment_plan", {**ARGS, "configuration": {**FIELDS, "prefillReplicas": 1}})
    assert payload["status"] == "invalid" and payload["validation"]["errors"][0]["code"] == "field_not_applicable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {**ARGS, "spec": {}},
        {**ARGS, "plan_id": "plan-x"},
        {**ARGS, "ttl_hours": 4},
        {**ARGS, "configuration": {**FIELDS, "replicas": -1}},
        {"cluster_id": "c1", "configuration": {**FIELDS, "topology": {}}},
    ],
)
async def test_plan_rejects_removed_and_invalid_arguments_before_any_request(api, ctx, arguments):
    payload, is_error = await call_tool(ctx, "deployment_plan", arguments)
    assert is_error and payload["error"]["code"] == "invalid_argument"
    assert api.calls == []


@pytest.mark.asyncio
async def test_plan_needs_exactly_one_source_and_a_cluster_for_a_configuration(api, ctx):
    for arguments in ({}, {**ARGS, "from_deployment_id": "run-1"}, {"configuration": FIELDS}):
        payload, is_error = await call_tool(ctx, "deployment_plan", arguments)
        assert is_error and payload["error"]["code"] == "invalid_argument"
    assert api.calls == []


@pytest.mark.asyncio
async def test_plan_needs_the_model_in_a_model_cache_volume_and_records_it(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "deployment_plan", ARGS)
    assert not is_error and payload["resolved_configuration"]["storageVolumeId"] == "storage-1"
    api.on("GET", "/model-cache/entries", {"items": []})
    payload, is_error = await call_tool(ctx, "deployment_plan", ARGS)
    assert not is_error and payload["status"] == "invalid"
    error = payload["validation"]["errors"][0]
    assert error["code"] == "model_not_cached" and "model_cache_create" in error["hint"]
    assert "modelPath" in error["hint"]


@pytest.mark.asyncio
async def test_plan_reports_a_misspelled_guide_before_judging_its_fields(api, ctx):
    plan_routes(api)
    configuration = {**FIELDS, "guide": "pd-disagregation", "prefillReplicas": 1}
    payload, _ = await call_tool(ctx, "deployment_plan", {**ARGS, "configuration": configuration})
    assert payload["validation"]["errors"][0]["code"] == "unknown_guide"


@pytest.mark.asyncio
async def test_plan_without_a_volume_names_the_public_field_and_the_path_preview(api, ctx):
    plan_routes(api)
    api.on("GET", "/storage/volumes", {"items": []})
    api.on("GET", "/model-cache/entries", {"items": []})
    payload, _ = await call_tool(ctx, "deployment_plan", ARGS)
    error = payload["validation"]["errors"][0]
    assert error["code"] == "prerequisite_missing" and error["field"] == "model_cache_volume_id"
    assert "set modelPath" in error["hint"]


@pytest.mark.asyncio
async def test_plan_with_an_explicit_model_path_says_the_path_is_unverified(api, ctx):
    plan_routes(api)
    configuration = {**FIELDS, "modelPath": "/models/Qwen/Qwen3-0.6B"}
    payload, is_error = await call_tool(ctx, "deployment_plan", {**ARGS, "configuration": configuration})
    assert not is_error and "not verified" in payload["next"] and "preview" in payload["next"]
    cached, _ = await call_tool(ctx, "deployment_plan", ARGS)
    assert cached["next"].startswith("Call deployment_create")


@pytest.mark.asyncio
async def test_plan_unknown_cluster_points_at_cluster_list(api, ctx):
    api.on("GET", "/cluster/overview", LensApiError(404, {"code": "cluster_not_found"}, "GET", "/cluster/overview"))
    payload, _ = await call_tool(ctx, "deployment_plan", {**ARGS, "cluster_id": "nope"})
    error = payload["error"]
    assert error["code"] == "not_found" and error["field"] == "cluster_id" and "nope" in error["message"]
    assert "cluster_list" in error["hint"]


# prefill/decode deployments need a reviewed plan ---------------------------------------------------------------

PD_ARGS = {"cluster_id": "c1", "configuration": FIELDS, "deployment_name": "pd-demo", "description": "pd"}
PD_PLAN = {
    "id": "plan-1",
    "digest": "sha256:abc",
    "preflight": {"status": "unknown", "errors": [], "warnings": [{"message": "tight"}], "unmodeled": ["DRA"]},
    "cases": [{"namespace": "pd-ns", "manifestChecksum": "m1", "manifest": "not for the agent"}],
}


PD_CONTENTS = [
    {"disaggregatedSet": {"slices": 1}},
    {"officialGuide": {"renderedManifest": "apiVersion: disaggregatedset.x-k8s.io/v1\nkind: DisaggregatedSet\n"}},
]


def pd_routes(api: FakeApi, plan=PD_PLAN, content=PD_CONTENTS[0]) -> None:
    plan_routes(api)
    api.on(
        "POST",
        "/configurations/render",
        lambda params, body: {"deployable_configuration": {"type": "baseline", "content": content}},
    )
    api.on("POST", "/deployments/plans", plan)
    api.on("POST", "/deployments/runs", {"id": "run-1"})


@pytest.mark.asyncio
@pytest.mark.parametrize("content", PD_CONTENTS)
async def test_plan_of_a_disaggregated_set_freezes_a_lens_plan_and_returns_its_review(api, ctx, content):
    pd_routes(api, content=content)
    payload, is_error = await call_tool(ctx, "deployment_plan", PD_ARGS)
    assert not is_error, payload
    assert payload["review"] == {
        "plan_id": "plan-1",
        "plan_digest": "sha256:abc",
        "preflight": {"status": "unknown", "errors": [], "warnings": ["tight"], "unknown_constraints": ["DRA"]},
        "workloads": [{"namespace": "pd-ns", "manifest_checksum": "m1"}],
    }
    assert "plan_id" in payload["next"] and "pd-demo" in payload["next"] and "ask them" in payload["next"]
    assert len(api.called("POST", "/deployments/plans")) == 1
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_plan_of_a_single_server_deployment_stores_nothing_in_lens(api, ctx):
    plan_routes(api)
    payload, _ = await call_tool(ctx, "deployment_plan", ARGS)
    assert "review" not in payload
    assert api.called("POST", "/deployments/plans") == []


@pytest.mark.asyncio
async def test_a_plan_whose_preflight_failed_is_invalid_and_not_creatable(api, ctx):
    failed = {**PD_PLAN, "preflight": {"status": "fail", "errors": [{"message": "no free roots"}], "unmodeled": []}}
    pd_routes(api, failed)
    payload, is_error = await call_tool(ctx, "deployment_plan", PD_ARGS)
    assert not is_error and payload["status"] == "invalid"
    assert payload["validation"]["errors"] == ["no free roots"]


@pytest.mark.asyncio
async def test_create_of_a_disaggregated_set_without_a_reviewed_plan_is_refused(api, ctx):
    pd_routes(api)
    payload, is_error = await call_tool(ctx, "deployment_create", PD_ARGS)
    assert is_error and payload["error"]["code"] == "plan_required"
    assert "deployment_plan" in payload["error"]["hint"]
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
async def test_create_sends_the_reviewed_plan_with_the_request_it_was_frozen_from(api, ctx):
    pd_routes(api)
    await call_tool(ctx, "deployment_plan", PD_ARGS)
    created, is_error = await call_tool(
        ctx,
        "deployment_create",
        {**PD_ARGS, "plan_id": "plan-1", "plan_digest": "sha256:abc", "acknowledge_unknown": True},
    )
    assert not is_error, created
    (frozen,) = api.called("POST", "/deployments/plans")
    (body,) = api.called("POST", "/deployments/runs")
    # Lens only executes a plan when the run request equals the one the plan was frozen from.
    assert {key: value for key, value in body.items() if key in frozen} == frozen
    assert body["prepared_plan_id"] == "plan-1"
    assert body["expected_plan_digest"] == "sha256:abc"
    assert body["acknowledge_unknown"] is True


@pytest.mark.asyncio
async def test_a_generated_name_can_be_passed_back_to_create_with_the_plan_digest(api, ctx):
    pd_routes(api)
    args = {key: value for key, value in PD_ARGS.items() if key != "deployment_name"}
    plan, is_error = await call_tool(ctx, "deployment_plan", args)
    assert not is_error, plan
    created, is_error = await call_tool(
        ctx,
        "deployment_create",
        {
            **args,
            "deployment_name": plan["name"],
            "expected_digest": plan["digest"],
            "plan_id": plan["review"]["plan_id"],
            "plan_digest": plan["review"]["plan_digest"],
        },
    )
    assert not is_error, created


@pytest.mark.asyncio
async def test_a_reviewed_plan_is_bound_to_the_deployment_name(api, ctx):
    pd_routes(api)
    args = {key: value for key, value in PD_ARGS.items() if key != "deployment_name"}
    payload, is_error = await call_tool(ctx, "deployment_create", {**args, "plan_id": "plan-1", "plan_digest": "d"})
    assert is_error and payload["error"]["field"] == "deployment_name"
    assert api.called("POST", "/deployments/runs") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Reviewed deployment plan digest does not match", "plan_stale"),
        ("Configuration, runtime binding or cluster changed; prepare a new plan", "plan_stale"),
        (
            "Deployment preflight failed or its unknown constraints have not been acknowledged",
            "unknown_not_acknowledged",
        ),
        ("something else", "plan_rejected"),
    ],
)
async def test_lens_refusing_a_reviewed_plan_tells_the_agent_what_to_do(api, ctx, message, code):
    pd_routes(api)
    api.on("POST", "/deployments/runs", LensApiError(409, {"detail": message}, "POST", "/deployments/runs"))
    payload, is_error = await call_tool(
        ctx, "deployment_create", {**PD_ARGS, "plan_id": "plan-1", "plan_digest": "sha256:abc"}
    )
    assert is_error and payload["error"]["code"] == code


@pytest.mark.asyncio
async def test_a_plan_id_is_refused_for_a_single_server_deployment(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "deployment_create", {**ARGS, "plan_id": "plan-1", "plan_digest": "d"})
    assert is_error and payload["error"]["code"] == "invalid_argument"
    assert api.called("POST", "/deployments/runs") == []


# create --------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sends_one_configuration_with_provenance_and_returns_the_run_id(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1"})
    created, is_error = await call_tool(ctx, "deployment_create", {**ARGS, "description": "for a test"})
    assert not is_error, created
    assert created["deployment_id"] == "run-1" and created["status"] == "pending"
    assert "deployment_get" in created["next"] and "wait_seconds=25" in created["next"]
    (body,) = api.called("POST", "/deployments/runs")
    assert len(body["configurations"]) == 1
    assert body["provenance"] == {
        "cluster_session_id": "s" * 36,
        "cluster_server_id": "c1",
        "created_via": "lens-mcp",
        "description": "for a test",
    }


@pytest.mark.asyncio
async def test_create_uses_the_requested_name_and_generates_one_otherwise(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1"})
    named, _ = await call_tool(ctx, "deployment_create", {**ARGS, "deployment_name": "demo"})
    assert named["name"] == "demo"
    generated, _ = await call_tool(ctx, "deployment_create", ARGS)
    assert generated["name"].startswith("mcp-qwen3-0-6b-")
    bad, is_error = await call_tool(ctx, "deployment_create", {**ARGS, "deployment_name": "Not Valid"})
    assert is_error and bad["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_digest_is_stable_across_plans_and_create_refuses_a_stale_one(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1"})
    first, _ = await call_tool(ctx, "deployment_plan", ARGS)
    second, _ = await call_tool(ctx, "deployment_plan", ARGS)
    assert first["digest"] == second["digest"]
    created, is_error = await call_tool(ctx, "deployment_create", {**ARGS, "expected_digest": first["digest"]})
    assert not is_error, created

    changed = {**ARGS, "configuration": {**FIELDS, "replicas": 2}}
    stale, is_error = await call_tool(ctx, "deployment_create", {**changed, "expected_digest": first["digest"]})
    assert is_error and stale["error"]["code"] == "plan_stale"
    assert stale["error"]["current_digest"] != first["digest"]
    assert len(api.called("POST", "/deployments/runs")) == 1


@pytest.mark.asyncio
async def test_create_refuses_a_digest_when_only_the_rendered_configuration_changed(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", {"id": "run-1"})

    def render(extra: list[str]):
        return lambda params, body: {
            "deployable_configuration": {
                "content": {"resources": ["server", *extra]},
                "checksum": f"sum-{len(extra)}-{body['render']['deploymentName']}",
            }
        }

    api.on("POST", "/configurations/render", render([]))
    first, _ = await call_tool(ctx, "deployment_plan", ARGS)
    again, _ = await call_tool(ctx, "deployment_plan", ARGS)
    assert first["digest"] == again["digest"]

    api.on("POST", "/configurations/render", render(["extra-service"]))
    stale, is_error = await call_tool(ctx, "deployment_create", {**ARGS, "expected_digest": first["digest"]})
    assert is_error and stale["error"]["code"] == "plan_stale"
    assert not api.called("POST", "/deployments/runs")


@pytest.mark.asyncio
async def test_create_rejected_by_lens_is_reported_and_nothing_else_is_written(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", LensApiError(422, {"detail": "bad manifest"}, "POST", "/deployments/runs"))
    payload, is_error = await call_tool(ctx, "deployment_create", ARGS)
    assert is_error and payload["error"]["code"] != "upstream_unavailable"


@pytest.mark.asyncio
async def test_create_with_an_uncertain_outcome_points_the_agent_to_the_listing(api, ctx):
    plan_routes(api)
    api.on("POST", "/deployments/runs", TimeoutError())
    payload, is_error = await call_tool(ctx, "deployment_create", ARGS)
    assert is_error and payload["error"]["code"] == "upstream_unavailable"
    assert "may or may not have completed" in payload["error"]["hint"]


@pytest.mark.asyncio
async def test_create_from_a_deployment_copies_its_configuration_under_a_fresh_name(api, ctx):
    plan_routes(api)
    configuration = {"type": "baseline", "content": {"customParameters": [{"name": "advanced"}]}, "provenance": {}}
    lens_state(api, configuration=configuration)
    api.on("POST", "/deployments/runs", {"id": "run-2"})
    created, is_error = await call_tool(ctx, "deployment_create", {"from_deployment_id": "run-1"})
    assert not is_error, created
    assert created["deployment_id"] == "run-2" and created["name"].startswith("mcp-clone-")
    (body,) = api.called("POST", "/deployments/runs")
    (sent,) = body["configurations"]
    assert sent["content"] == configuration["content"]
    assert sent["provenance"]["deployment_name"] == created["name"]
    assert configuration["provenance"] == {}  # the stored configuration is not mutated
    assert api.called("POST", "/guide-planning/plan") == []


@pytest.mark.asyncio
async def test_a_copy_keeps_the_source_cluster_and_storage_binding(api, ctx):
    plan_routes(api)
    lens_state(api, configuration={"content": {"storage": "cluster-bound"}})
    other_cluster, error = await call_tool(ctx, "deployment_plan", {"from_deployment_id": "run-1", "cluster_id": "c2"})
    assert error and other_cluster["error"]["field"] == "cluster_id"
    volume, error = await call_tool(
        ctx, "deployment_plan", {"from_deployment_id": "run-1", "model_cache_volume_id": "storage-1"}
    )
    assert error and volume["error"]["field"] == "model_cache_volume_id"
    assert api.called("GET", "/cluster/session") == []


@pytest.mark.asyncio
async def test_a_copy_of_a_deployment_without_stored_configuration_is_invalid(api, ctx):
    plan_routes(api)
    lens_state(api)
    payload, is_error = await call_tool(ctx, "deployment_plan", {"from_deployment_id": "run-1"})
    assert not is_error and payload["status"] == "invalid"
    assert payload["validation"]["errors"][0]["code"] == "configuration_unavailable"


@pytest.mark.asyncio
async def test_a_copy_of_an_unknown_deployment_is_not_found(api, ctx):
    plan_routes(api)
    payload, is_error = await call_tool(ctx, "deployment_create", {"from_deployment_id": "run-x"})
    assert is_error and payload["error"]["code"] == "deployment_not_found"


# deployment_get ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_deployment_reports_roles_endpoint_and_spec(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(
        api,
        status="deploying",
        replicas=2,
        model="Qwen/Qwen3-0.6B",
        guide="optimized-baseline",
        endpoint="http://gw",
        pods=[
            pod("p1", "decode"),
            pod("p2", "decode", ready=False),
            pod("w", None),
            {**pod("epp", None), "labels": {"llm-d-router-gateway": "x"}},
        ],
    )
    got, is_error = await call_tool(ctx, "deployment_get", {"deployment_id": deployment_id})
    assert not is_error, got
    assert got["deployment_id"] == "run-1" and got["execution_id"] == EXEC and got["name"] == "demo"
    assert got["status"] == "starting" and got["ready"] is False and got["owned"] is True
    assert got["ownership"] == "self" and got["capabilities"]["stop"] is True
    assert got["roles"]["decode"] == {"ready": 1, "total": 2, "desired": 2}
    assert "router" in got["roles"] and len(got["roles"]) == 2  # the harness pod is not part of the deployment
    assert got["spec"]["model"] == "Qwen/Qwen3-0.6B" and got["endpoint"] == "http://gw" and got["observed_at"]
    assert "expires_at" not in got


@pytest.mark.asyncio
async def test_get_deployment_returns_the_stored_lens_configuration(api, ctx):
    configuration = {"type": "baseline", "content": {"maxNumSeqs": 32}}
    lens_state(api, configuration=configuration)
    got, is_error = await call_tool(ctx, "deployment_get", {"deployment_id": "run-1"})
    assert not is_error and got["configuration"] == configuration, got


@pytest.mark.asyncio
async def test_get_deployment_by_execution_id_works_and_reports_ownership_from_lens(api, ctx):
    lens_state(
        api, pods=[pod("p1", "decode")], model="m", guide="optimized-baseline", ownership="other", capabilities={}
    )
    got, is_error = await call_tool(ctx, "deployment_get", {"deployment_id": EXEC})
    assert not is_error and got["owned"] is False and got["ownership"] == "other" and got["status"] == "ready"
    assert got["deployment_id"] == "run-1" and got["spec"]["model"] == "m"
    api.routes.pop(("GET", f"/deployments/executions/{EXEC}"))
    missing, is_error = await call_tool(ctx, "deployment_get", {"deployment_id": EXEC})
    assert is_error and missing["error"]["code"] == "deployment_not_found"


@pytest.mark.asyncio
async def test_get_deployment_of_a_run_without_execution_reads_the_case_status_and_failure(api, ctx):
    api.on(
        "GET",
        "/deployments/runs/run-1",
        {"id": "run-1", "cases": [{**lens_case("failed", None), "failure": {"message": "no capacity"}}]},
    )
    got, is_error = await call_tool(ctx, "deployment_get", {"deployment_id": "run-1"})
    assert not is_error, got
    assert got["status"] == "failed" and got["owned"] is False and got["ownership"] == "unknown"
    assert got["untrusted_text"] == [{"source": "lens_failure", "lines": ["no capacity"]}]


@pytest.mark.asyncio
async def test_get_deployment_long_poll_returns_on_status_change(api, ctx, monkeypatch):
    monkeypatch.setattr(observe, "POLL_SECONDS", 0.01)
    lens_state(api, status="queued")
    reads = {"n": 0}

    def execution(params, body):
        reads["n"] += 1
        return {"execution_id": EXEC, "status": "queued" if reads["n"] < 3 else "ready", "namespace": "ns"}

    api.on("GET", f"/deployments/executions/{EXEC}", execution)
    got, _ = await call_tool(ctx, "deployment_get", {"deployment_id": "run-1", "wait_seconds": 5})
    assert got["status"] == "ready" and reads["n"] == 3


@pytest.mark.asyncio
async def test_a_run_with_several_cases_needs_the_exact_execution_id(api, ctx):
    api.on(
        "GET",
        "/deployments/runs/run-1",
        {
            "id": "run-1",
            "cases": [
                {"id": "case-a", "ordinal": 0, "attempt": 1, "execution_id": "exec-a"},
                {"id": "case-b", "ordinal": 1, "attempt": 2, "execution_id": "exec-b"},
            ],
        },
    )
    with pytest.raises(ToolError, match="multiple deployment cases"):
        await deployment_refs.resolve(ctx, "run-1")


@pytest.mark.asyncio
async def test_the_newest_attempt_of_a_case_chain_is_the_deployment(api, ctx):
    lens_state(api, status="deploying", execution_id="exec-2", case_status="deploying")
    api.on(
        "GET",
        "/deployments/runs/run-1",
        {"id": "run-1", "cases": [lens_case("stopped", EXEC), lens_case("deploying", "exec-2", attempt=2)]},
    )
    got, _ = await call_tool(ctx, "deployment_get", {"deployment_id": "run-1"})
    assert got["execution_id"] == "exec-2" and got["status"] == "starting"


# ownership, stop and delete -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_posts_to_the_latest_case_and_is_idempotent(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("POST", STOP, {})
    stopped, is_error = await call_tool(ctx, "deployment_stop", {"deployment_id": deployment_id})
    assert not is_error and stopped["status"] == "stopped"
    # The Lens route is POST; any other method on this path is a bug.
    assert [method for method, path, _ in api.calls if path == STOP] == ["POST"]
    lens_state(api, status="rolled_back", case_status="stopped")
    again, _ = await call_tool(ctx, "deployment_stop", {"deployment_id": deployment_id})
    assert again["already_stopped"] is True
    assert len(api.called("POST", STOP)) == 1


@pytest.mark.asyncio
async def test_stop_by_execution_id_targets_the_exact_case_despite_a_newer_sibling(api, ctx):
    lens_state(api, run_id="run-web", case_id="case-web")
    api.on("POST", "/deployments/runs/run-web/cases/case-web/stop", {})
    result, error = await call_tool(ctx, "deployment_stop", {"deployment_id": EXEC})
    assert not error and result["status"] == "stopped", result
    assert len(api.called("POST", "/deployments/runs/run-web/cases/case-web/stop")) == 1


@pytest.mark.asyncio
async def test_stop_conflict_from_lens_is_actionable(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("POST", STOP, LensApiError(409, {"detail": "no active cluster session"}, "POST", STOP))
    payload, is_error = await call_tool(ctx, "deployment_stop", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "conflict_state" and payload["error"]["hint"]


@pytest.mark.asyncio
async def test_stop_reports_a_consumer_still_using_the_deployment(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("POST", STOP, LensApiError(409, {"code": "deployment_in_use", "detail": "active benchmark"}, "POST", STOP))
    result, error = await call_tool(ctx, "deployment_stop", {"deployment_id": deployment_id})
    assert error and result["error"]["code"] == "in_use", result


@pytest.mark.asyncio
async def test_lens_policy_denial_stops_a_deployment_the_agent_may_not_change(api, ctx):
    lens_state(api, ownership="other", capabilities={"update": False, "stop": False, "delete": False})
    result, error = await call_tool(ctx, "deployment_stop", {"deployment_id": "run-1"})
    assert error and result["error"]["code"] == "not_owner"
    # Owning the deployment is not enough when Lens withholds the capability.
    lens_state(api, ownership="self", capabilities={"stop": False})
    result, error = await call_tool(ctx, "deployment_stop", {"deployment_id": "run-1"})
    assert error and result["error"]["code"] == "owner_unknown"
    assert api.called("POST", STOP) == []


@pytest.mark.asyncio
async def test_a_deployment_without_lens_ownership_is_not_changed(api, ctx):
    lens_state(api, ownership="unknown", capabilities={})
    payload, is_error = await call_tool(ctx, "deployment_stop", {"deployment_id": EXEC})
    assert is_error and payload["error"]["code"] == "owner_unknown"
    api.on("GET", "/deployments/runs/run-1", {"id": "run-1", "cases": [lens_case("queued", None)]})
    payload, is_error = await call_tool(ctx, "deployment_stop", {"deployment_id": "run-1"})
    assert is_error and payload["error"]["code"] == "owner_unknown"


def delete_args(name="demo", deployment_id="run-1"):
    return {"items": [{"kind": "deployment", "id": deployment_id, "name": name}]}


@pytest.mark.asyncio
async def test_delete_removes_the_execution_and_its_namespace(api, ctx):
    deployment_id = await create(api, ctx)
    api.on("DELETE", f"/deployments/executions/{EXEC}", {})
    payload, is_error = await call_tool(ctx, "resource_delete", delete_args(deployment_id=deployment_id))
    assert not is_error, payload
    assert payload["results"][0]["status"] == "deleted"
    assert api.called("DELETE", f"/deployments/executions/{EXEC}") == [{"delete_namespace": True}]


@pytest.mark.asyncio
async def test_delete_with_a_stale_name_refuses_before_deleting(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, display_name="Renamed in Web")
    result, error = await call_tool(ctx, "resource_delete", delete_args(deployment_id=deployment_id))
    assert error and result["error"]["code"] == "id_mismatch", result
    assert not any(method == "DELETE" for method, _, _ in api.calls)


@pytest.mark.asyncio
async def test_delete_of_another_users_deployment_is_refused_per_item(api, ctx):
    lens_state(api, ownership="other", capabilities={})
    payload, _ = await call_tool(ctx, "resource_delete", delete_args())
    (result,) = payload["results"]
    assert result["status"] == "failed" and result["error"] == "not_owner"
    assert not any(method == "DELETE" for method, _, _ in api.calls)


@pytest.mark.asyncio
async def test_a_missing_execution_is_reported_deleted_and_never_deletes_the_parent_run(api, ctx):
    deployment_id = await create(api, ctx)  # no DELETE route for the execution: Lens answers 404
    api.on("DELETE", "/deployments/runs/run-1", {})
    payload, is_error = await call_tool(ctx, "resource_delete", delete_args(deployment_id=deployment_id))
    assert not is_error, payload
    assert payload["results"][0]["status"] == "deleted"
    assert api.called("DELETE", "/deployments/runs/run-1") == []


@pytest.mark.asyncio
async def test_delete_of_an_unknown_deployment_is_not_found(api, ctx):
    payload, is_error = await call_tool(ctx, "resource_delete", delete_args(deployment_id="run-x"))
    assert not is_error and payload["results"][0]["status"] == "not_found"


@pytest.mark.asyncio
async def test_delete_in_use_deployment_is_blocked_not_deleted(api, ctx):
    deployment_id = await create(api, ctx)
    api.on(
        "DELETE",
        f"/deployments/executions/{EXEC}",
        LensApiError(409, {"code": "deployment_in_use", "detail": "used by evaluation"}, "DELETE", "x"),
    )
    payload, _ = await call_tool(ctx, "resource_delete", delete_args(deployment_id=deployment_id))
    (result,) = payload["results"]
    assert result["status"] == "blocked" and result["error"] == "in_use"


@pytest.mark.asyncio
async def test_delete_409_without_code_is_a_blocked_conflict_with_a_hint(api, ctx):
    deployment_id = await create(api, ctx)
    api.on(
        "DELETE",
        f"/deployments/executions/{EXEC}",
        LensApiError(409, {"detail": "deployment is still running"}, "DELETE", "x"),
    )
    payload, _ = await call_tool(ctx, "resource_delete", delete_args(deployment_id=deployment_id))
    (result,) = payload["results"]
    assert result["status"] == "blocked" and result["error"] == "conflict_state"
    assert "deployment_stop" in result["message"]


@pytest.mark.asyncio
async def test_resource_list_uses_lens_ownership_and_a_run_id_when_it_names_one_execution(api, ctx):
    def execution(execution_id, ownership, run_id, status="ready"):
        return {
            "execution_id": execution_id,
            "run_id": run_id,
            "status": status,
            "display_name": execution_id,
            "cluster_id": "c1",
            "ownership": ownership,
        }

    api.on("GET", "/cluster/clusters", {"items": [{"id": "c1"}]})
    items = [
        execution(EXEC, "self", "run-1"),
        execution("exec-b", "other", "run-b"),
        execution("exec-z", "unknown", None),
        execution("exec-old", "self", "run-old", "cleaned"),
        execution("exec-m1", "self", "run-m"),
        execution("exec-m2", "self", "run-m"),
    ]
    api.on("GET", "/deployments/executions", {"items": items})
    payload, is_error = await call_tool(
        ctx, "resource_list", {"kinds": ["deployment"], "owner": "all", "cluster_id": "c1"}
    )
    assert not is_error, payload
    ownership = {row["id"]: row["ownership"] for row in payload["resources"]}
    assert ownership == {
        "run-1": "owned",
        "run-b": "shared",
        "exec-z": "unknown",
        "exec-m1": "owned",
        "exec-m2": "owned",
    }
    mine, _ = await call_tool(ctx, "resource_list", {"kinds": ["deployment"]})
    assert {row["id"] for row in mine["resources"]} == {"run-1", "exec-m1", "exec-m2"}


# diagnose ------------------------------------------------------------------------------------------------------


def logs(api, pod_name, text):
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
    result, is_error = await call_tool(ctx, "deployment_diagnose", {"deployment_id": deployment_id})
    assert not is_error, result
    codes = {f["code"] for f in result["findings"]}
    assert {"image_pull_error", "oom_killed", "crash_loop", "unschedulable"} <= codes
    assert result["healthy"] is False
    assert all("Insufficient" not in str(f) for f in result["findings"])
    assert any("Insufficient" in line for entry in result["untrusted_text"] for line in entry["lines"])


@pytest.mark.asyncio
async def test_diagnose_finds_model_download_failure_in_logs_and_capacity_shortage(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, status="deploying", replicas=30, tensor_parallel_size=1, pods=[pod("a", "decode", ready=False)])
    logs(api, "a", "huggingface_hub.errors.GatedRepoError: 401 Client Error. Access to model is restricted")
    result, _ = await call_tool(ctx, "deployment_diagnose", {"deployment_id": deployment_id})
    codes = {f["code"] for f in result["findings"]}
    assert "model_access_denied" in codes
    assert "insufficient_capacity" not in codes  # all pods exist and are scheduled
    lens_state(
        api,
        status="queued",
        replicas=30,
        tensor_parallel_size=1,
        pods=[pod("a", "decode", ready=False, phase="Pending", reason="Pending")],
    )
    result, _ = await call_tool(ctx, "deployment_diagnose", {"deployment_id": deployment_id})
    assert "insufficient_capacity" in {f["code"] for f in result["findings"]}


@pytest.mark.asyncio
async def test_diagnose_healthy_deployment_has_no_findings(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("a", "decode")])
    result, _ = await call_tool(ctx, "deployment_diagnose", {"deployment_id": deployment_id})
    assert result["healthy"] is True and result["findings"] == []


# deployment_logs -----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_logs_defaults_to_unhealthy_pod_and_caps_text(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("good", "decode"), pod("bad", "decode", ready=False, restarts=2)])
    logs(api, "bad", "x" * 30000)
    result, is_error = await call_tool(ctx, "deployment_logs", {"deployment_id": deployment_id})
    assert not is_error, result
    assert result["pod"] == "bad" and result["container"] == "vllm" and result["truncated"] is True
    assert len(result["untrusted_text"]) == 20000
    assert result["other_pods"] == [{"name": "good", "role": "decode", "ready": True}]


@pytest.mark.asyncio
async def test_get_logs_selects_by_role_and_rejects_unknown_pod_or_container(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, pods=[pod("pre", "prefill"), pod("dec", "decode")])
    logs(api, "pre", "prefill output")
    result, _ = await call_tool(
        ctx, "deployment_logs", {"deployment_id": deployment_id, "role": "prefill", "previous": True}
    )
    assert result["pod"] == "pre" and result["untrusted_text"] == "prefill output"
    assert api.called("GET", f"/deployments/executions/{EXEC}/pods/pre/logs")[-1]["previous"] is True
    for args in ({"pod": "nope"}, {"role": "router"}, {"container": "nope"}):
        payload, is_error = await call_tool(ctx, "deployment_logs", {"deployment_id": deployment_id, **args})
        assert is_error and payload["error"]["code"] in ("not_found", "invalid_argument")


# deployment_test -----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_deployment_returns_shared_lens_record(api, ctx):
    deployment_id = await create(api, ctx)
    record = {
        "id": "verification-1",
        "execution_id": EXEC,
        "status": "succeeded",
        "verdict": "inconclusive",
        "requests_sent": 10,
        "checks": [{"check": "pd_transfer", "verdict": "inconclusive"}],
    }
    path = f"/deployments/executions/{EXEC}/verifications"
    api.on("POST", path, record)
    result, is_error = await call_tool(
        ctx, "deployment_test", {"deployment_id": deployment_id, "checks": ["chat", "prefix_cache"], "requests": 10}
    )
    assert not is_error, result
    assert result["verdict"] == "inconclusive" and result["verification_id"] == "verification-1"
    assert result["checks"] == record["checks"]
    assert api.called("POST", path) == [{"checks": ["chat", "prefix_cache"], "requests": 10}]
    assert not api.called("POST", f"/deployments/executions/{EXEC}/endpoint")


@pytest.mark.asyncio
async def test_test_deployment_pending_result_can_be_read_without_replay(api, ctx):
    lens_state(api)
    path = f"/deployments/executions/{EXEC}/verifications"
    api.on("POST", path, {"id": "verification-1", "status": "queued"})
    started, error = await call_tool(ctx, "deployment_test", {"deployment_id": EXEC, "wait_seconds": 0})
    assert not error and started["status"] == "queued", started
    api.on("GET", f"{path}/verification-1", {"id": "verification-1", "status": "succeeded", "verdict": "pass"})
    got, error = await call_tool(ctx, "deployment_test", {"deployment_id": EXEC, "verification_id": "verification-1"})
    assert not error and got["verdict"] == "pass", got
    assert len(api.called("POST", path)) == 1
    api.on("GET", path, {"items": [{"id": "verification-1", "status": "succeeded", "verdict": "pass"}]})
    history, error = await call_tool(ctx, "deployment_get", {"deployment_id": EXEC, "include": ["verifications"]})
    assert not error, history
    assert history["verifications"][0]["id"] == "verification-1"
    assert len(api.called("POST", path)) == 1


@pytest.mark.asyncio
async def test_test_deployment_prefix_cache_validates_total_budget(api, ctx):
    deployment_id = await create(api, ctx)
    args = {"deployment_id": deployment_id, "checks": ["prefix_cache"], "requests": 10}
    short, is_error = await call_tool(ctx, "deployment_test", {**args, "requests": 3})
    assert is_error and short["error"]["field"] == "requests"


@pytest.mark.asyncio
async def test_test_deployment_is_limited_to_deployments_the_caller_owns(api, ctx):
    lens_state(api, ownership="other", capabilities={})
    path = f"/deployments/executions/{EXEC}/verifications"
    api.on("POST", path, {"id": "verification-1", "status": "queued"})
    result, error = await call_tool(ctx, "deployment_test", {"deployment_id": EXEC})
    assert error and result["error"]["code"] == "not_owner"
    assert not api.called("POST", path)
    api.on("GET", f"{path}/verification-1", {"id": "verification-1", "status": "succeeded", "verdict": "pass"})
    got, error = await call_tool(ctx, "deployment_test", {"deployment_id": EXEC, "verification_id": "verification-1"})
    assert not error and got["verdict"] == "pass"


@pytest.mark.asyncio
async def test_test_deployment_honors_lens_permission_revocation(api, ctx):
    lens_state(api)
    path = f"/deployments/executions/{EXEC}/verifications"
    api.on("POST", path, LensApiError(403, {"detail": "permission revoked"}, "POST", path))
    result, error = await call_tool(ctx, "deployment_test", {"deployment_id": EXEC})
    assert error, result
    assert not api.called("POST", f"/deployments/executions/{EXEC}/endpoint")


@pytest.mark.asyncio
async def test_test_deployment_requires_ready_deployment_and_at_most_20_requests(api, ctx):
    deployment_id = await create(api, ctx)
    lens_state(api, status="deploying", pods=[pod("d1", "decode", ready=False)])
    path = f"/deployments/executions/{EXEC}/verifications"
    api.on("POST", path, LensApiError(409, {"detail": "not ready"}, "POST", path))
    payload, is_error = await call_tool(ctx, "deployment_test", {"deployment_id": deployment_id})
    assert is_error and payload["error"]["code"] == "conflict_state"
    payload, is_error = await call_tool(ctx, "deployment_test", {"deployment_id": deployment_id, "requests": 21})
    assert is_error and payload["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_resource_list_rejects_an_unknown_cluster_instead_of_listing_nothing(api, ctx):
    api.on("GET", "/cluster/clusters", {"items": [{"id": "c1"}]})
    payload, is_error = await call_tool(ctx, "resource_list", {"cluster_id": "nope"})
    assert is_error and payload["error"]["code"] == "not_found" and payload["error"]["field"] == "cluster_id"


@pytest.mark.asyncio
async def test_invalid_arguments_report_every_problem(api, ctx):
    payload, is_error = await call_tool(ctx, "model_cache_create", {})
    error = payload["error"]
    assert is_error and error["code"] == "invalid_argument"
    assert {item["field"] for item in error["all_problems"]} == {"model", "cluster_id"}


@pytest.mark.asyncio
async def test_resource_list_explains_an_empty_result(api, ctx):
    api.on("GET", "/evaluate/workflow-runs", {"items": []})
    payload, _ = await call_tool(ctx, "resource_list", {"kinds": ["experiment"]})
    assert payload["count"] == 0 and "owner='all'" in payload["empty_note"]


@pytest.mark.asyncio
async def test_diagnose_does_not_blame_capacity_for_a_pod_that_is_already_placed(api, ctx):
    deployment_id = await create(api, ctx)
    placed = {**pod("a", "decode", ready=False, phase="Pending", reason="ContainerCreating"), "node_name": "smc-19"}
    lens_state(api, status="deploying", pods=[placed])
    result, _ = await call_tool(ctx, "deployment_diagnose", {"deployment_id": deployment_id})
    codes = {f["code"] for f in result["findings"]}
    assert codes == {"pod_starting"}
    assert "Free accelerators" not in str(result["findings"])
