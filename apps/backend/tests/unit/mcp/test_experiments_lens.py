"""Experiments on Lens Evaluation workflows. Request bodies are validated with the real Lens models."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from llm_d_lens.configuration.router import RenderRequest, SaveRequest
from llm_d_lens.deploy.contracts import DeployableConfiguration
from llm_d_lens.evaluation.models import EvaluationCreateRequest
from llm_d_lens.evaluation.router import _evaluation_cases, _ordered_evaluation_cases
from llm_d_lens.mcp import evaluation_ops, evaluation_view
from llm_d_lens.mcp.errors import ToolError
from llm_d_lens.mcp.evaluation_design import ExperimentDesign, evaluation_request, expand, predicted_cases
from llm_d_lens.mcp.evaluation_plan import _baseline_gpus
from llm_d_lens.mcp.guide_fields import GuidePlanFields, normalize_fields
from llm_d_lens.mcp.lens_api import LensApiError
from llm_d_lens.mcp.planning import VariantPlan
from llm_d_lens.mcp.registry import all_tools, call_tool

from .backend_routes import record_route_problems
from .conftest import FakeApi, load_plan, plan_routes

WF = "/evaluate/workflow-runs"


def metrics(ttft_p99: float, tps: float) -> dict:
    return {
        "throughput_tps": tps,
        "throughput_rps": tps / 128,
        "success_rate": 100.0,
        "request_count": 32,
        "latency_distributions": {
            "ttft": {"p50_ms": ttft_p99 / 2, "p90_ms": ttft_p99 * 0.8, "p99_ms": ttft_p99},
            "tpot": {"p50_ms": 20.0, "p99_ms": 40.0},
            "request_latency": {"p50_ms": 3000.0, "p99_ms": 4000.0},
        },
    }


def succeeded_run(run_id: str, ttft_p99: float = 800.0, tps: float = 500.0, model: str = "Qwen/Qwen3-0.6B") -> dict:
    return {
        "id": run_id,
        "status": "succeeded",
        "cluster_id": "c1",
        "configuration": {"model": model},
        "hardware_profile": {"id": "intel-xpu"},
        "benchmark_runtime": {"commit": "abc123"},
        "matrix_results": [
            {
                "isl": 512,
                "osl": 128,
                "status": "succeeded",
                "stage_metrics": [
                    {"concurrency": 1, "num_requests": 32, "metrics": metrics(ttft_p99 / 4, tps / 4)},
                    {"concurrency": 4, "num_requests": 32, "metrics": metrics(ttft_p99, tps)},
                ],
            }
        ],
    }


def test_annotations_follow_the_delta_table():
    tools = {spec.name: spec for spec in all_tools()}
    for name in ("experiment_plan", "experiment_get", "result_compare"):
        assert tools[name].read_only and tools[name].idempotent and not tools[name].destructive
    assert not tools["experiment_start"].read_only and not tools["experiment_start"].destructive
    assert tools["experiment_start"].open_world
    cancel = tools["experiment_cancel"]
    assert cancel.destructive and cancel.idempotent and not cancel.read_only
    assert not tools["experiment_retry"].read_only and not tools["experiment_retry"].destructive
    confirm = tools["experiment_confirm"]
    assert not confirm.read_only and not confirm.destructive and confirm.profile == "core"


SCENARIO = {"id": "chat", "name": "Chat", "benchmark": {"matrix": [{"isl": 512, "osl": 128}]}}
SCENARIO_B = {"id": "long", "name": "Long", "benchmark": {"matrix": [{"isl": 4096, "osl": 128}]}}
COMMON = {"guide": "optimized-baseline", "model": "Qwen/Qwen3-0.6B"}
DESIGN = {
    "name": "ab",
    "cluster_id": "c1",
    "common": COMMON,
    "variants": [{"id": "base"}, {"id": "tp2", "plan": {"tensorParallelSize": 2}}],
    "scenarios": [SCENARIO],
}


def uuid(number: int) -> str:
    return f"{number:08d}-0000-4000-8000-000000000000"


class Evaluations:
    """Fake Lens evaluation API; creation builds the cases with Lens' own case builder."""

    def __init__(self, api: FakeApi):
        self.api = api
        self.workflows: dict[str, dict] = {}
        self.details: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.saved = 0
        api.on("POST", "/evaluate/timing-estimate", {"lower_seconds": 60, "upper_seconds": 120})
        api.on("POST", "/configurations/render", self.render)
        api.on("POST", "/configurations/save", self.save)
        api.on("POST", "/evaluate/evaluations", self.create)
        api.on("GET", "/evaluate/workflow-runs", lambda *_: {"items": list(self.workflows.values())})

    def render(self, params, body) -> dict:
        RenderRequest.model_validate(body)
        checksum = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return {
            "deployable_configuration": {
                "type": body["candidate_config"]["type"],
                "format": "yaml",
                "content": {},
                "provider_ref": "optimized-baseline",
                "checksum": checksum,
            },
            "metadata": {},
        }

    def save(self, params, body) -> dict:
        SaveRequest.model_validate(body)
        self.saved += 1
        artifact_id = uuid(self.saved)
        self.api.on("DELETE", f"/configurations/artifacts/{artifact_id}", {})
        return {"artifact": {"artifact_id": artifact_id}}

    def create(self, params, body) -> dict:
        request = EvaluationCreateRequest.model_validate(body)
        artifacts = {
            plan.configuration_artifact_id: SimpleNamespace(
                deployable_configuration=DeployableConfiguration(
                    type="baseline", format="yaml", provider_ref="optimized-baseline", checksum="x"
                )
            )
            for plan in request.benchmark_plans
        }
        cases = _ordered_evaluation_cases(_evaluation_cases(request, artifacts))
        workflow_id = uuid(1000 + len(self.workflows))
        workflow = {
            "id": workflow_id,
            "kind": "workflow",
            "name": request.name,
            "status": "queued",
            "created_at": "2026-05-01T10:00:00+00:00",
            "cluster_id": "c1",
            "owner_user_id": self.api.owner,
            "benchmark_plans": [plan.model_dump(mode="json") for plan in request.benchmark_plans],
            "cases": [{**case, "deployment_configuration": None} for case in cases],
        }
        self.add(workflow)
        return workflow

    def add(self, workflow: dict) -> dict:
        workflow_id = workflow["id"]
        self.workflows[workflow_id] = workflow
        base = f"/evaluate/workflow-runs/{workflow_id}"
        self.api.on("GET", base, lambda *_: self.workflows[workflow_id])
        self.api.on("GET", f"{base}/details", lambda *_: self.details[workflow_id])
        self.api.on("POST", f"{base}/cancel", lambda *_: self.cancel(workflow_id))
        self.api.on("POST", f"{base}/retry", lambda *_: {**self.workflows[workflow_id], "status": "queued"})
        self.api.on("DELETE", base, {})
        return workflow

    def cancel(self, workflow_id: str) -> dict:
        self.workflows[workflow_id]["status"] = "cancelled"
        return {"id": workflow_id, "status": "cancelled"}

    def run(self, run_id: str, run: dict) -> None:
        self.runs[run_id] = run
        self.api.on("GET", f"/evaluate/runs/{run_id}", lambda *_: self.runs[run_id])


@pytest.fixture
def backend_routes(api):
    problems = record_route_problems(api)
    yield problems
    assert problems == []


@pytest.fixture
def lens(api, backend_routes) -> Evaluations:
    plan_routes(api)
    return Evaluations(api)


async def run(ctx, tool_name: str, /, **arguments):
    return await call_tool(ctx, tool_name, arguments)


def design(**overrides) -> ExperimentDesign:
    return ExperimentDesign.model_validate({**DESIGN, **overrides})


def foreign(lens: Evaluations, status: str = "running", owner: str | None = "someone-else") -> str:
    workflow = lens.add(
        {
            "id": uuid(7000 + len(lens.workflows)),
            "name": "theirs",
            "status": status,
            "cluster_id": "c1",
            "owner_user_id": owner,
            "created_at": "2026-05-01T10:00:00+00:00",
            "cases": [{"id": "guide-1-1", "kind": "guide", "status": "running" if status == "running" else status}],
        }
    )
    return workflow["id"]


@pytest.mark.asyncio
async def test_plan_is_stable_validated_and_writes_nothing(api, ctx, lens):
    first, is_error = await run(ctx, "experiment_plan", **DESIGN)
    assert not is_error, first
    second, _ = await run(ctx, "experiment_plan", **DESIGN)
    assert first["request_digest"] == second["request_digest"] and first["seed"] == second["seed"]
    assert first["cases_total"] == 2 and first["peak_gpus"] == 2 and first["deployments_total"] == 2
    assert first["eta_seconds"]["benchmark"]["upper"] > 0 and first["status"] in ("valid", "valid_with_warnings")
    assert [item["variant_id"] for item in first["plans"]] == ["base", "tp2"]
    assert api.called("POST", "/configurations/save") == [] and api.called("POST", "/evaluate/evaluations") == []
    changed, _ = await run(ctx, "experiment_plan", **{**DESIGN, "scenarios": [SCENARIO_B]})
    assert changed["request_digest"] != first["request_digest"]


@pytest.mark.asyncio
async def test_plan_budgets_an_independent_baseline_by_its_own_size(api, ctx, lens):
    design_with_baseline = {
        **DESIGN,
        "variants": [{"id": "base"}],
        "baseline_types": ["direct-vllm", "kubernetes-service"],
        "baseline_parameters": {"direct-vllm": {"replicas": 1, "tensor_parallel_size": 8}},
    }
    plan, is_error = await run(ctx, "experiment_plan", **design_with_baseline)
    assert not is_error, plan
    assert plan["deployments_total"] == 2 and plan["peak_gpus"] == 8
    kept, _ = await run(ctx, "experiment_plan", **{**design_with_baseline, "preserve_deployment": True})
    assert kept["peak_gpus"] == 9  # the kept guide deployment (1) plus the baseline (8)


def test_a_pd_baseline_budget_that_does_not_divide_is_rejected_at_plan_time():
    pd_content = {
        "prefill": {"replicaCount": 1, "tensorParallelSize": 1},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1},
    }
    configuration = {
        "type": "baseline",
        "format": "helm",
        "provider_ref": "pd-disaggregation",
        "checksum": "x",
        "content": pd_content,
    }
    variants = {"base": VariantPlan("base", {}, "xpu", "vllm", "base", "img", 2, configuration, [], None, [])}
    ok = design(variants=[{"id": "base"}], baseline_types=["direct-vllm"])
    assert _baseline_gpus(ok, variants) == {("base", "direct-vllm"): 2}  # 2 GPUs of budget at TP1
    bad = design(
        variants=[{"id": "base"}],
        baseline_types=["direct-vllm"],
        baseline_parameters={"direct-vllm": {"tensor_parallel_size": 4}},
    )
    with pytest.raises(ToolError) as caught:
        _baseline_gpus(bad, variants)
    assert caught.value.code == "plan_invalid" and "baseline" in str(caught.value)


@pytest.mark.asyncio
async def test_plan_says_when_the_benchmark_time_cannot_be_estimated(api, ctx, lens):
    api.on("POST", "/evaluate/timing-estimate", {"basis": "unknown", "lower_seconds": None, "upper_seconds": None})
    payload, is_error = await run(ctx, "experiment_plan", **DESIGN)
    assert not is_error, payload
    times = payload["eta_seconds"]
    assert times["benchmark"] == {"lower": None, "upper": None} and times["deploy"] > 0
    assert "matrix" in times["unknown"] and times["how_to_check"]
    assert payload["gpu_hours_estimate"] is None
    assert payload["cases_breakdown"]["guide"] == 2 and payload["not_checked"]


@pytest.mark.asyncio
async def test_plan_warns_about_shared_deployment_repetitions_and_counts_cases(api, ctx, lens):
    payload, is_error = await run(ctx, "experiment_plan", **{**DESIGN, "repetitions": {"count": 3}})
    assert not is_error, payload
    assert payload["cases_total"] == 6
    assert "REPEATS_SHARE_DEPLOYMENT" in {warning["code"] for warning in payload["warnings"]}


@pytest.mark.asyncio
async def test_plan_names_the_variant_whose_guide_is_unknown(api, ctx, lens):
    design = {**DESIGN, "variants": [{"id": "base"}, {"id": "typo", "plan": {"guide": "optimzed-baseline"}}]}
    payload, is_error = await run(ctx, "experiment_plan", **design)
    error = payload["validation"]["errors"][0]
    assert not is_error and error["code"] == "unknown_guide"
    assert error["field"] == "variants/typo/plan/guide" and error["message"].startswith("Variant 'typo'")


@pytest.mark.asyncio
async def test_plan_with_an_explicit_model_path_says_the_path_is_unverified(api, ctx, lens):
    payload, _ = await run(ctx, "experiment_plan", **DESIGN)
    assert payload["next"].startswith("Call experiment_start")
    design = {**DESIGN, "common": {**COMMON, "modelPath": "/models/Qwen/Qwen3-0.6B"}}
    payload, is_error = await run(ctx, "experiment_plan", **design)
    assert not is_error and "not verified" in payload["next"]


@pytest.mark.asyncio
async def test_plan_rejects_inapplicable_and_oversized_designs(api, ctx, lens):
    pd_only = {**DESIGN, "variants": [{"id": "base", "plan": {"prefillReplicas": 1}}]}
    payload, is_error = await run(ctx, "experiment_plan", **pd_only)
    assert not is_error and payload["status"] == "invalid"
    assert payload["validation"]["errors"][0]["code"] == "field_not_applicable"
    many = {**DESIGN, "scenarios": [{**SCENARIO, "id": f"s{index}"} for index in range(11)]}
    payload, is_error = await run(ctx, "experiment_plan", **{**many, "repetitions": {"count": 2}})
    assert not is_error and payload["validation"]["errors"][0]["code"] == "limit_exceeded"
    payload, is_error = await run(ctx, "experiment_start", **{**many, "repetitions": {"count": 2}})
    assert is_error and payload["error"]["code"] == "limit_exceeded"
    assert api.called("POST", "/guide-planning/plan") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["experiment_plan", "experiment_start"])
@pytest.mark.parametrize(
    "benchmark",
    [
        {"parallelism": 1.5},
        {"parallelism": True},
        {"warmup_requests": "2"},
        {"matrix": [{"isl": 0, "osl": 128}]},
        {"matrix": [{"isl": 512.5, "osl": 128}]},
        {"concurrency_stages": [{"concurrency": 1, "num_requests": 2.5}]},
        {"concurrency_stages": "not-an-array"},
        {"shared_prefix": "not-an-object"},
        "not-an-object",
    ],
)
async def test_invalid_workload_dimensions_never_reach_lens(api, ctx, tool_name, benchmark):
    arguments = {**DESIGN, "scenarios": [{"id": "invalid", "name": "Invalid", "benchmark": benchmark}]}
    payload, is_error = await run(ctx, tool_name, **arguments)
    assert is_error and payload["error"]["code"] in {"invalid_argument", "limit_exceeded"}
    assert payload["error"]["field"].startswith("scenarios/0/benchmark")
    assert api.calls == []


def test_case_view_preserves_native_lens_resource_relationships():
    case = {
        "id": "case-1",
        "evaluation_run_id": "evaluation-1",
        "deployment_run_id": "deployment-1",
        "deployment_execution_id": "execution-1",
        "deployment_ownership": "evaluation-owned",
        "namespace": "deployment-namespace",
    }
    view = evaluation_view.case_view(case)
    assert view["run_id"] == view["evaluation_run_id"] == "evaluation-1"
    for key in ("deployment_run_id", "deployment_execution_id", "deployment_ownership", "namespace"):
        assert view[key] == case[key]
    unknown = evaluation_view.case_view({"id": "case-2"})
    assert "namespace" not in unknown and "deployment_execution_id" not in unknown


def test_experiment_schema_exposes_native_benchmark_and_baseline_contracts():
    from llm_d_lens.evaluation.models import BenchmarkScenario, BenchmarkSpec

    schema = ExperimentDesign.model_json_schema()
    assert schema["properties"]["scenarios"]["items"]["$ref"].endswith("/BenchmarkScenario")
    assert schema["$defs"]["BenchmarkSpec"]["properties"] == BenchmarkSpec.model_json_schema()["properties"]
    assert {"matrix", "concurrency_stages", "shared_prefix", "workload_yaml", "harness_memory_gib"} <= set(
        schema["$defs"]["BenchmarkSpec"]["properties"]
    )
    parsed = design()
    assert isinstance(parsed.scenarios[0], BenchmarkScenario)
    assert expand(parsed, 0)[0].scenarios == [SCENARIO]


@pytest.mark.asyncio
@pytest.mark.parametrize("guide", ["optimized-baseline", "pd-disaggregation"])
async def test_shared_configuration_entry_matches_experiment_requests(api, ctx, lens, guide):
    catalog = api.routes[("GET", "/guide-planning/catalog")]
    catalog["guides"].append({**catalog["guides"][0], "id": "pd-disaggregation"})
    plan = load_plan()
    plan["source"]["guide"] = guide
    api.on("POST", "/guide-planning/plan", plan)
    fields = {
        **COMMON,
        "guide": guide,
        "maxModelLen": 4096,
        "maxNumSeqs": 64,
        "customParameters": [{"target": "both", "kind": "argument", "name": "enforce-eager", "value": "true"}],
        "guideSettings": {"routerValues": "test: true"},
    }
    if guide == "pd-disaggregation":
        fields.update(prefillReplicas=1, prefillTensorParallelSize=2, decodeReplicas=2, decodeTensorParallelSize=2)
    single, error = await run(
        ctx, "deployment_plan", cluster_id="c1", deployment_name="same-name", configuration=fields
    )
    assert not error and single["status"] in {"valid", "valid_with_warnings"}, single
    experiment, error = await run(
        ctx,
        "experiment_plan",
        **{
            **DESIGN,
            "common": fields,
            "variants": [{"id": "base", "deployment_name": "same-name"}],
        },
    )
    assert not error, experiment
    first, second = api.called("POST", "/guide-planning/plan")
    assert first == second
    first, second = api.called("POST", "/configurations/render")
    assert first == second
    assert single["resolved_configuration"]["maxNumSeqs"] == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["deployment_plan", "experiment_plan"])
@pytest.mark.parametrize(
    "source",
    [
        {"mode": "official"},
        {"mode": "local", "path": "/allowed/custom/kustomization.yaml"},
        {"mode": "remote", "repository": "example/guides", "path": "guides/custom.yaml", "ref": "release"},
    ],
)
async def test_source_and_yaml_patches_reach_native_planning_and_render(api, ctx, lens, monkeypatch, source, tool_name):
    from llm_d_lens.configuration import service
    from llm_d_lens.deploy.guide_planning import apply_custom_patches

    plan = load_plan()
    plan.pop("deploymentBundle", None)
    patch = {
        "path": "placement.yaml",
        "content": yaml.safe_dump(
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "optimized-baseline-xpu-vllm-decode"},
                "spec": {"template": {"spec": {"nodeSelector": {"pool": "test-workers"}}}},
            }
        ),
    }

    def planned(params, body):
        assert body["source"] == source or body["source"] == {**source, "ref": "main"}
        assert body["customPatches"] == [patch]
        documents = list(yaml.safe_load_all(plan["plannedDeployment"]["content"]))
        errors, warnings = [], []
        applied = apply_custom_patches(documents, body["customPatches"], errors, warnings)
        assert not errors
        return {
            **plan,
            "plannedDeployment": {"content": yaml.safe_dump_all(documents)},
            "appliedCustomPatches": applied,
        }

    api.on("POST", "/guide-planning/plan", planned)
    if source["mode"] != "official":
        api.on("GET", "/guide-planning/catalog", LensApiError(409, {}, "GET", "/guide-planning/catalog"))
    monkeypatch.setattr(service, "require_active_session", lambda _: SimpleNamespace(server_id="c1"))
    contents = []

    def render(params, body):
        response = service.render_configuration(RenderRequest.model_validate(body))
        contents.append(response.deployable_configuration.content)
        return response.model_dump(mode="json")

    api.on("POST", "/configurations/render", render)
    fields = {**COMMON, "modelSource": "huggingface", "source": source, "customPatches": [patch]}
    arguments = (
        {"cluster_id": "c1", "configuration": fields}
        if tool_name == "deployment_plan"
        else {
            **DESIGN,
            "common": fields,
            "variants": [{"id": "base"}],
        }
    )
    payload, error = await run(ctx, tool_name, **arguments)
    assert not error and payload["status"] in {"valid", "valid_with_warnings"}, payload
    result = payload if tool_name == "deployment_plan" else payload["plans"][0]
    assert result["source"] == plan["source"]
    assert result["applied_custom_patches"][0]["action"] == "merged"
    if source["mode"] != "official":
        assert api.called("GET", "/guide-planning/catalog") == []
    (content,) = contents
    documents = list(yaml.safe_load_all(content["officialGuide"]["renderedManifest"]))
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    assert deployment["spec"]["template"]["spec"]["nodeSelector"]["pool"] == "test-workers"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["deployment_plan", "deployment_create", "experiment_plan", "experiment_start"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"source": {"mode": "remote", "repository": "https://elsewhere.test/owner/repo", "path": "a.yaml"}},
        {"source": {"mode": "remote", "repository": "owner/repo", "path": "../a.yaml"}},
        {"source": {"mode": "remote", "repository": "owner/repo", "path": "a.yaml", "ref": "bad ref"}},
        {"source": {"mode": "local"}},
        {"source": {"mode": "unknown"}},
        {"customPatches": [{"content": "kind: ["}]},
        {"customPatches": [{"content": "- not-a-resource"}]},
        {"customPatches": [{"content": "kind: Deployment\nmetadata: invalid"}]},
        {"customPatches": [{"content": "#" + "é" * 131072}]},
        {"customPatches": [{"content": "kind: ConfigMap"}] * 21},
    ],
)
async def test_invalid_source_and_patches_are_rejected_before_lens(api, ctx, tool_name, invalid):
    fields = {**COMMON, **invalid}
    arguments = (
        {"cluster_id": "c1", "configuration": fields}
        if "deployment" in tool_name
        else {
            **DESIGN,
            "common": fields,
        }
    )
    payload, error = await run(ctx, tool_name, **arguments)
    assert error and payload["error"]["code"] == "invalid_argument", payload
    assert api.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, "shared-path"])
async def test_experiment_model_path_survives_native_configuration_render(api, ctx, lens, monkeypatch, source):
    from llm_d_lens.configuration import service
    from llm_d_lens.configuration.manifest_facts import validate_manifest_facts

    path = "/models/shared/qwen"
    model = COMMON["model"]
    manifest = yaml.safe_dump(
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "modelserver", "labels": {"llm-d.ai/role": "decode"}},
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "modelserver",
                                "image": "registry.example/vllm:test",
                                "command": ["vllm", "serve"],
                                "args": ["/model-cache", "--served-model-name", model],
                                "volumeMounts": [{"name": "model", "mountPath": "/model-cache", "readOnly": True}],
                            }
                        ],
                        "volumes": [{"name": "model", "hostPath": {"path": path}}],
                    }
                },
            },
        }
    )
    plan = load_plan()
    plan["plannedDeployment"]["content"] = manifest
    plan["deployment"]["readinessDeployments"] = ["modelserver"]
    plan.pop("deploymentBundle", None)
    api.on("POST", "/guide-planning/plan", plan)
    monkeypatch.setattr(service, "require_active_session", lambda _: SimpleNamespace(server_id="c1"))
    rendered = []

    def render(params, body):
        result = service.render_configuration(RenderRequest.model_validate(body))
        rendered.append(result.deployable_configuration.content)
        return result.model_dump(mode="json")

    api.on("POST", "/configurations/render", render)
    common = {**COMMON, "modelPath": path}
    if source:
        common["modelSource"] = source
    payload, is_error = await run(ctx, "experiment_plan", **{**DESIGN, "common": common, "variants": [{"id": "base"}]})
    assert not is_error, payload
    assert api.called("GET", "/model-cache/entries") == []
    (request,) = api.called("POST", "/guide-planning/plan")
    assert request["modelPath"] == path
    (content,) = rendered
    assert content["runtime"]["mountPath"] == path
    assert content["runtime"]["modelSource"] == "shared-path"
    mismatched = manifest.replace(path, "/models/different")
    with pytest.raises(ValueError, match="storage host path"):
        validate_manifest_facts(content, mismatched)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["experiment_plan", "experiment_start"])
@pytest.mark.parametrize(
    "scenario",
    [
        {**SCENARIO, "id": "invalid/id"},
        {**SCENARIO, "name": "x" * 201},
        {**SCENARIO, "benchmark": {"harness_memory_gib": 0}},
        {**SCENARIO, "benchmark": {"shared_prefix": {"num_groups": 1}}},
        {**SCENARIO, "benchmark": {"workload_yaml": "load: {}"}},
        {**SCENARIO, "sla_targets": {"success_rate_min_percent": 101}},
        {**SCENARIO, "benchmark": {"concurrency": [1, 2]}},
        {**SCENARIO, "benchmark": {"matrix": [{"isl": 8, "osl": 8, "typo": 1}]}},
        {**SCENARIO, "slo": {}},
    ],
)
async def test_native_scenario_validation_precedes_all_lens_calls(api, ctx, tool_name, scenario):
    payload, is_error = await run(ctx, tool_name, **{**DESIGN, "scenarios": [scenario]})
    assert is_error and payload["error"]["code"] == "invalid_argument"
    assert api.calls == []


@pytest.mark.asyncio
async def test_native_baseline_parameters_reject_invalid_values_before_planning(api, ctx):
    payload, is_error = await run(
        ctx,
        "experiment_plan",
        **{
            **DESIGN,
            "baseline_parameters": {"direct-vllm": {"tensor_parallel_size": 0}},
        },
    )
    assert is_error and payload["error"]["code"] == "invalid_argument"
    assert api.calls == []


@pytest.mark.asyncio
async def test_unknown_baseline_parameter_key_is_rejected_with_allowed_keys(api, ctx):
    payload, is_error = await run(
        ctx, "experiment_plan", **{**DESIGN, "baseline_parameters": {"direct-vllm": {"tp": 2}}}
    )
    assert is_error and payload["error"]["code"] == "invalid_argument"
    assert "tp" in payload["error"]["message"] and "tensor_parallel_size" in payload["error"]["hint"]
    assert api.calls == []


def test_pd_fields_conflict_when_aliases_disagree():
    fields = GuidePlanFields.model_validate(
        {**COMMON, "guide": "pd-disaggregation", "replicas": 2, "decodeReplicas": 3}
    )
    with pytest.raises(ToolError) as error:
        normalize_fields(fields.model_dump(by_alias=True, exclude_none=True), "v")
    assert error.value.code == "field_conflict"


def test_same_deployment_repetitions_interleave_scenarios_on_one_plan():
    spec = design(scenarios=[SCENARIO, SCENARIO_B], repetitions={"count": 2})
    plans = expand(spec, 0)
    assert [plan.plan_id for plan in plans] == ["base", "tp2"]
    assert [item["id"] for item in plans[0].scenarios] == ["chat-r1", "long-r1", "chat-r2", "long-r2"]
    assert plans[0].scenarios[2]["name"] == "Chat (r2/2)"


def test_fresh_deployment_repetitions_copy_the_plan_and_shuffle_is_seeded():
    spec = design(repetitions={"count": 3, "mode": "fresh_deployment"}, order="shuffled", seed=7)
    plans = expand(spec, 7)
    assert sorted(plan.plan_id for plan in plans) == ["base-r1", "base-r2", "base-r3", "tp2-r1", "tp2-r2", "tp2-r3"]
    assert [p.plan_id for p in plans] == [p.plan_id for p in expand(spec, 7)]
    assert [p.plan_id for p in plans] != [p.plan_id for p in expand(spec, 8)]
    artifacts = {"base": uuid(1), "tp2": uuid(2)}
    body = evaluation_request(spec, plans, artifacts, "s" * 36, 7, "n")
    request = EvaluationCreateRequest.model_validate(body)
    assert all("lens-mcp variant=" in plan.deployment_description for plan in request.benchmark_plans)
    assert not any(plan.include_baseline for plan in request.benchmark_plans)


def test_predicted_case_order_matches_the_lens_case_builder():
    spec = design(scenarios=[SCENARIO, SCENARIO_B], baseline_types=["direct-vllm", "kubernetes-service", "load-only"])
    plans = expand(spec, 0)
    body = evaluation_request(spec, plans, {"base": uuid(1), "tp2": uuid(2)}, "s" * 36, 0, "n")
    request = EvaluationCreateRequest.model_validate(body)
    artifact = SimpleNamespace(
        deployable_configuration=DeployableConfiguration(
            type="baseline", format="yaml", provider_ref="optimized-baseline", checksum="x"
        )
    )
    real = _ordered_evaluation_cases(_evaluation_cases(request, {uuid(1): artifact, uuid(2): artifact}))
    assert [case["id"] for case in predicted_cases(spec, plans)] == [case["id"] for case in real]


@pytest.mark.asyncio
async def test_start_saves_one_artifact_per_variant_and_returns_without_advancing(api, ctx, lens):
    plan, _ = await run(ctx, "experiment_plan", **DESIGN)
    started, is_error = await run(ctx, "experiment_start", **DESIGN, expected_digest=plan["request_digest"])
    assert not is_error, started
    assert len(api.called("POST", "/configurations/save")) == 2 and lens.saved == 2
    (workflow,) = lens.workflows.values()
    assert started["experiment_id"] == workflow["id"] and started["status"] == "queued"
    assert started["cases_total"] == 2 and started["poll_after_seconds"] and "experiment_get" in started["next"]
    assert [case["id"] for case in started["case_order"]] == [case["id"] for case in workflow["cases"]]
    assert {call[0:2] for call in api.calls if call[0] != "GET" and call[1].startswith("/evaluate")} == {
        ("POST", "/evaluate/evaluations"),
        ("POST", "/evaluate/timing-estimate"),
    }
    assert not any(path.endswith("/cancel") or "/deployments" in path for _, path, _ in api.calls)


@pytest.mark.asyncio
async def test_start_refuses_a_stale_digest_before_saving_anything(api, ctx, lens):
    payload, is_error = await run(ctx, "experiment_start", **DESIGN, expected_digest="sha256:stale")
    assert is_error and payload["error"]["code"] == "plan_stale" and payload["error"]["current_digest"]
    assert api.called("POST", "/configurations/save") == [] and lens.workflows == {}


@pytest.mark.asyncio
async def test_failed_create_removes_the_artifacts_it_saved(api, ctx, lens):
    api.on(
        "POST",
        "/evaluate/evaluations",
        LensApiError(409, {"detail": "Insufficient accelerator capacity for this evaluation"}, "POST", "x"),
    )
    payload, is_error = await run(ctx, "experiment_start", **DESIGN)
    assert is_error and payload["error"]["code"] == "insufficient_capacity"
    assert payload["error"]["leftover_artifacts"] == []
    deleted = [path for method, path, _ in api.calls if method == "DELETE"]
    assert sorted(deleted) == sorted(f"/configurations/artifacts/{uuid(n)}" for n in (1, 2))


PD_PLANS = [
    {
        "case_id": "guide-1-1",
        "id": "plan-1",
        "digest": "sha256:abc",
        "preflight": {"status": "unknown", "errors": [], "warnings": [], "unmodeled": ["DRA"]},
        "cases": [{"namespace": "pd-ns", "manifestChecksum": "m1"}],
    }
]


def awaiting_review(lens: Evaluations, owner: str = "user-a") -> str:
    workflow = lens.add(
        {
            "id": uuid(8000 + len(lens.workflows)),
            "name": "pd",
            "status": "awaiting_review",
            "cluster_id": "c1",
            "owner_user_id": owner,
            "created_at": "2026-05-01T10:00:00+00:00",
            "prepared_plans": PD_PLANS,
            "cases": [{"id": "guide-1-1", "kind": "guide", "status": "queued", "benchmark": {"matrix": []}}],
        }
    )
    lens.api.on(
        "POST",
        f"/evaluate/evaluations/{workflow['id']}/confirm",
        lambda params, body: {"id": workflow["id"], "status": "queued"},
    )
    return workflow["id"]


@pytest.mark.asyncio
async def test_plan_tells_the_agent_that_a_prefill_decode_design_waits_for_review(api, ctx, lens):
    plain, _ = await run(ctx, "experiment_plan", **DESIGN)
    assert "awaiting_review" not in plain["next"]
    catalog = api.routes[("GET", "/guide-planning/catalog")]
    catalog["guides"].append({**catalog["guides"][0], "id": "pd-disaggregation"})
    plan = load_plan()
    plan["source"]["guide"] = "pd-disaggregation"
    api.on("POST", "/guide-planning/plan", plan)
    pd_design = {
        **DESIGN,
        "variants": [{"id": "pd"}],
        "common": {
            **COMMON,
            "guide": "pd-disaggregation",
            "prefillReplicas": 1,
            "prefillTensorParallelSize": 1,
            "decodeReplicas": 1,
            "decodeTensorParallelSize": 1,
        },
    }
    planned, is_error = await run(ctx, "experiment_plan", **pd_design)
    assert not is_error, planned
    assert "awaiting_review" in planned["next"] and "experiment_confirm" in planned["next"]


@pytest.mark.asyncio
async def test_start_reports_an_experiment_that_waits_for_plan_review(api, ctx, lens):
    create = lens.create
    api.on(
        "POST",
        "/evaluate/evaluations",
        lambda params, body: {**create(params, body), "status": "awaiting_review", "prepared_plans": PD_PLANS},
    )
    started, is_error = await run(ctx, "experiment_start", **DESIGN)
    assert not is_error, started
    assert started["status"] == "awaiting_review" and "poll_after_seconds" not in started
    (plan,) = started["review_required"]["plans"]
    assert plan["plan_id"] == "plan-1" and plan["plan_digest"] == "sha256:abc" and plan["case_id"] == "guide-1-1"
    assert plan["preflight"]["unknown_constraints"] == ["DRA"]
    assert "experiment_confirm" in started["next"] and "review" in started["next"]


@pytest.mark.asyncio
async def test_get_shows_awaiting_review_as_its_own_phase_and_not_as_stale(api, ctx, lens, monkeypatch):
    monkeypatch.setattr(evaluation_view, "now", lambda: evaluation_view.parse_time("2026-05-02T10:00:00+00:00"))
    experiment_id = awaiting_review(lens)
    view, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id)
    assert not is_error, view
    assert view["status"] == "awaiting_review" and view["poll_after_seconds"] is None
    assert view["stale"] == {"is_stale": False}
    assert view["review_required"]["plans"][0]["plan_id"] == "plan-1"
    assert "experiment_confirm" in view["review_required"]["next"]


@pytest.mark.asyncio
async def test_confirm_sends_the_reviewed_digests_and_queues_the_experiment(api, ctx, lens):
    experiment_id = awaiting_review(lens)
    done, is_error = await run(
        ctx,
        "experiment_confirm",
        experiment_id=experiment_id,
        plan_digests={"plan-1": "sha256:abc"},
        acknowledge_unknown=True,
    )
    assert not is_error, done
    assert done["status"] == "queued" and "experiment_get" in done["next"]
    (body,) = api.called("POST", f"/evaluate/evaluations/{experiment_id}/confirm")
    assert body == {"plan_digests": {"plan-1": "sha256:abc"}, "acknowledge_unknown": True}


@pytest.mark.asyncio
async def test_confirm_is_limited_to_the_creator_and_to_experiments_awaiting_review(api, ctx, lens):
    theirs = awaiting_review(lens, owner="someone-else")
    refused, is_error = await run(ctx, "experiment_confirm", experiment_id=theirs, plan_digests={"plan-1": "d"})
    assert is_error and refused["error"]["code"] == "not_owner"
    running = running_workflow(lens)
    refused, is_error = await run(ctx, "experiment_confirm", experiment_id=running, plan_digests={"plan-1": "d"})
    assert is_error and refused["error"]["code"] == "conflict_state"
    assert api.called("POST", f"/evaluate/evaluations/{theirs}/confirm") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Reviewed Evaluation plan digests do not match", "plan_stale"),
        ("Prepared Evaluation plan is unavailable", "plan_stale"),
        ("Preflight failed or unknown constraints were not acknowledged", "unknown_not_acknowledged"),
    ],
)
async def test_confirm_refused_by_lens_tells_the_agent_what_to_do(api, ctx, lens, message, code):
    experiment_id = awaiting_review(lens)
    api.on(
        "POST",
        f"/evaluate/evaluations/{experiment_id}/confirm",
        LensApiError(409, {"detail": message}, "POST", "x"),
    )
    refused, is_error = await run(ctx, "experiment_confirm", experiment_id=experiment_id, plan_digests={"plan-1": "d"})
    assert is_error and refused["error"]["code"] == code


def running_workflow(lens: Evaluations, owner: str = "user-a", heartbeat: str = "2026-05-01T10:10:00+00:00") -> str:
    workflow = lens.add(
        {
            "id": uuid(9000 + len(lens.workflows)),
            "name": "mine",
            "status": "benchmarking",
            "cluster_id": "c1",
            "owner_user_id": owner,
            "created_at": "2026-05-01T10:00:00+00:00",
            "started_at": "2026-05-01T10:01:00+00:00",
            "active_case_id": "guide-1-1",
            "cases": [
                {"id": "guide-1-1", "kind": "guide", "status": "benchmarking", "evaluation_run_id": "ev-1"},
                {"id": "guide-2-1", "kind": "guide", "status": "queued", "benchmark": {"matrix": []}},
            ],
        }
    )
    lens.run("ev-1", {"id": "ev-1", "status": "running", "phase": "running", "heartbeat_at": heartbeat})
    return workflow["id"]


@pytest.mark.asyncio
async def test_get_reports_progress_and_never_reads_details_unless_asked(api, ctx, lens, monkeypatch):
    now = evaluation_view.parse_time("2026-05-01T10:10:30+00:00")
    monkeypatch.setattr(evaluation_view, "now", lambda: now)
    experiment_id = running_workflow(lens)
    view, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id)
    assert not is_error, view
    assert view["progress"]["queued"] == 1 and view["progress"]["active"] == 1
    assert view["active_case"]["case_id"] == "guide-1-1" and view["stale"] == {"is_stale": False}
    assert view["last_progress_at"]["value"] == "2026-05-01T10:10:00+00:00" and view["poll_after_seconds"] >= 15
    assert api.called("GET", f"/evaluate/workflow-runs/{experiment_id}/details") == []
    assert "waited_seconds" not in view
    lens.runs["ev-1"]["heartbeat_at"] = "2026-05-01T10:05:00+00:00"
    stale, _ = await run(ctx, "experiment_get", experiment_id=experiment_id)
    assert stale["stale"]["is_stale"] and stale["stale"]["reason"] and stale["stale"]["hint"]


@pytest.mark.asyncio
async def test_get_really_waits_until_the_experiment_changes(api, ctx, lens, monkeypatch):
    experiment_id = running_workflow(lens)
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) == 2:
            lens.workflows[experiment_id]["cases"][0]["status"] = "succeeded"
            lens.workflows[experiment_id]["cases"][1]["status"] = "benchmarking"

    monkeypatch.setattr(evaluation_view, "sleep", fake_sleep)
    view, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id, wait_seconds=40)
    assert not is_error, view
    assert slept == [5, 5] and view["wait_outcome"] == "changed" and view["waited_seconds"] >= 10
    assert view["progress"]["succeeded"] == 1


@pytest.mark.asyncio
async def test_get_wait_times_out_and_terminal_wait_ignores_progress(api, ctx, lens, monkeypatch):
    experiment_id = running_workflow(lens)
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) == 1:
            lens.workflows[experiment_id]["cases"][0]["status"] = "succeeded"

    monkeypatch.setattr(evaluation_view, "sleep", fake_sleep)
    view, _ = await run(ctx, "experiment_get", experiment_id=experiment_id, wait_seconds=10, wait_for="terminal")
    assert view["wait_outcome"] == "timeout" and sum(slept) == 10
    over, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id, wait_seconds=51)
    assert is_error and over["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_get_results_and_comparison_use_details_once(api, ctx, lens):
    experiment_id = running_workflow(lens)
    lens.workflows[experiment_id]["status"] = "succeeded"
    lens.workflows[experiment_id]["benchmark_plans"] = [
        {"id": "base", "deployment_description": "lens-mcp variant=base rep=1/1 order=as_listed seed=none"},
        {"id": "tp2", "deployment_description": "lens-mcp variant=tp2 rep=1/1 order=as_listed seed=none"},
    ]
    lens.details[experiment_id] = {
        "workflow": lens.workflows[experiment_id],
        "cases": [
            {
                "case": {"id": f"guide-{n}-1", "kind": "guide", "benchmark_plan_id": plan, "scenario_id": "chat"},
                "evaluation": succeeded_run(f"ev-{n}", ttft_p99=800.0 * n, tps=500.0 / n),
            }
            for n, plan in ((1, "base"), (2, "tp2"))
        ],
    }
    view, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id, include=["comparison", "cases"])
    assert not is_error, view
    assert len(api.called("GET", f"/evaluate/workflow-runs/{experiment_id}/details")) == 1
    assert {item["plan"] for item in view["results"]} == {"base", "tp2"} and len(view["cases"]) == 2
    row = next(r for r in view["comparison"]["rows"] if r["plan"] == "tp2" and r["metric"] == "ttft_ms.p99")
    assert row["delta_vs_baseline_pct"] == 100.0
    compared, is_error = await run(ctx, "result_compare", experiment_ids=[experiment_id], baseline="tp2")
    assert not is_error, compared
    assert compared["comparison"]["baseline"] == "tp2"
    bad, is_error = await run(ctx, "result_compare", experiment_ids=[experiment_id], baseline="nope")
    assert is_error and bad["error"]["field"] == "baseline"


@pytest.mark.asyncio
async def test_unknown_experiment_is_not_found(api, ctx, lens):
    missing, is_error = await run(ctx, "experiment_get", experiment_id=uuid(4242))
    assert is_error and missing["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_artifacts_are_listed_and_per_request_records_are_summarised(api, ctx, lens):
    experiment_id = running_workflow(lens)
    api.on(
        "GET",
        "/evaluate/runs/ev-1/artifacts",
        {"items": [{"name": "per_request_lifecycle_metrics.json", "size_bytes": 99}], "archived_in": []},
    )
    listing, is_error = await run(ctx, "experiment_artifact_get", experiment_id=experiment_id, case_id="guide-1-1")
    assert not is_error, listing
    assert listing["run_id"] == "ev-1" and listing["items"][0]["size_bytes"] == 99
    records = [{"ttft_ms": float(value), "ok": True, "id": "r"} for value in range(1, 101)]
    api.on("GET", "/evaluate/runs/ev-1/artifacts/per_request_lifecycle_metrics.json", records)
    summary, is_error = await run(
        ctx,
        "experiment_artifact_get",
        experiment_id=experiment_id,
        case_id="guide-1-1",
        name="per_request_lifecycle_metrics.json",
    )
    assert not is_error, summary
    assert summary["records"] == 100 and len(summary["sample"]) == 3
    assert summary["numeric_fields"]["ttft_ms"]["p50"] == 51.0 and summary["numeric_fields"]["ttft_ms"]["max"] == 100.0
    assert "ok" not in summary["numeric_fields"]
    assert summary["download_path"] == "/api/v1/evaluate/runs/ev-1/artifacts/per_request_lifecycle_metrics.json"


@pytest.mark.asyncio
async def test_artifacts_need_a_known_case_that_has_started_and_a_clean_name(api, ctx, lens):
    experiment_id = running_workflow(lens)
    payload, is_error = await run(ctx, "experiment_artifact_get", experiment_id=experiment_id, case_id="nope")
    assert is_error and payload["error"]["field"] == "case_id"
    payload, is_error = await run(ctx, "experiment_artifact_get", experiment_id=experiment_id, case_id="guide-2-1")
    assert is_error and payload["error"]["code"] == "conflict_state"
    payload, is_error = await run(
        ctx, "experiment_artifact_get", experiment_id=experiment_id, case_id="guide-1-1", name="a/../b.json"
    )
    assert is_error and payload["error"]["field"] == "name"
    payload, is_error = await run(
        ctx, "experiment_artifact_get", experiment_id=experiment_id, case_id="guide-1-1", name="missing.json"
    )
    assert is_error and payload["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_cancel_is_limited_to_the_creator_and_reports_cleanup(api, ctx, lens):
    theirs = foreign(lens)
    refused, is_error = await run(ctx, "experiment_cancel", experiment_id=theirs)
    assert (
        is_error
        and refused["error"]["code"] == "not_owner"
        and api.called("POST", f"/evaluate/workflow-runs/{theirs}/cancel") == []
    )
    unknown = foreign(lens, owner=None)
    refused, _ = await run(ctx, "experiment_cancel", experiment_id=unknown)
    assert refused["error"]["code"] == "owner_unknown"
    mine = running_workflow(lens)
    done, is_error = await run(ctx, "experiment_cancel", experiment_id=mine, reason="wrong model")
    assert not is_error, done
    assert done["status"] == "cancelled" and done["cleanup"]["state"] == "done"
    again, _ = await run(ctx, "experiment_cancel", experiment_id=mine)
    assert again["outcome"] == "already_done" and len(api.called("POST", f"/evaluate/workflow-runs/{mine}/cancel")) == 1


@pytest.mark.asyncio
async def test_a_cancel_that_outlasts_the_request_timeout_is_a_retryable_cleanup(api, ctx, lens):
    mine = running_workflow(lens)
    api.on("POST", f"/evaluate/workflow-runs/{mine}/cancel", TimeoutError())
    payload, is_error = await run(ctx, "experiment_cancel", experiment_id=mine)
    assert is_error and payload["error"]["code"] == "cleanup_pending" and payload["error"]["retryable"]


@pytest.mark.asyncio
async def test_cancel_case_and_cleanup_failure(api, ctx, lens):
    mine = running_workflow(lens)
    unknown, is_error = await run(ctx, "experiment_cancel", experiment_id=mine, case_ids=["nope"])
    assert is_error and unknown["error"]["field"] == "case_ids"
    api.on("POST", f"/evaluate/workflow-runs/{mine}/cases/guide-2-1/cancel", lambda *_: {"id": "guide-2-1"})
    single, is_error = await run(ctx, "experiment_cancel", experiment_id=mine, case_ids=["guide-2-1"])
    assert not is_error, single
    assert api.called("POST", f"/evaluate/workflow-runs/{mine}/cancel") == []
    api.on(
        "POST", f"/evaluate/workflow-runs/{mine}/cancel", LensApiError(409, {"detail": "cleanup failed"}, "POST", "x")
    )
    pending, is_error = await run(ctx, "experiment_cancel", experiment_id=mine)
    assert is_error and pending["error"]["code"] == "cleanup_pending" and pending["error"]["retryable"]


@pytest.mark.asyncio
async def test_retry_only_failed_or_cancelled_experiments_of_the_creator(api, ctx, lens):
    running = running_workflow(lens)
    refused, is_error = await run(ctx, "experiment_retry", experiment_id=running)
    assert is_error and refused["error"]["code"] == "conflict_state"
    lens.workflows[running]["status"] = "failed"
    lens.workflows[running]["cases"][0]["status"] = "failed"
    done, is_error = await run(ctx, "experiment_retry", experiment_id=running)
    assert not is_error, done
    assert done["status"] == "queued" and done["requeued_case_ids"] == ["guide-1-1"]
    theirs = foreign(lens, status="failed")
    other, _ = await run(ctx, "experiment_retry", experiment_id=theirs)
    assert other["error"]["code"] == "not_owner"


@pytest.mark.asyncio
async def test_a_retry_that_needs_new_plans_comes_back_for_review(api, ctx, lens):
    failed = running_workflow(lens)
    lens.workflows[failed]["status"] = "failed"
    lens.workflows[failed]["cases"][0]["status"] = "failed"
    api.on(
        "POST",
        f"/evaluate/workflow-runs/{failed}/retry",
        lambda *_: {**lens.workflows[failed], "status": "awaiting_review", "prepared_plans": PD_PLANS},
    )
    payload, is_error = await run(ctx, "experiment_retry", experiment_id=failed)
    assert not is_error, payload
    assert payload["status"] == "awaiting_review" and payload["requeued_case_ids"] == ["guide-1-1"]
    assert "poll_after_seconds" not in payload
    assert [plan["plan_id"] for plan in payload["review_required"]["plans"]] == [PD_PLANS[0]["id"]]


@pytest.mark.asyncio
async def test_delete_removes_terminal_experiments_and_their_artifacts(api, ctx, lens):
    started, _ = await run(ctx, "experiment_start", **DESIGN)
    experiment_id = started["experiment_id"]
    active, _ = await run(ctx, "resource_delete", items=[{"kind": "experiment", "id": experiment_id, "name": "ab"}])
    assert (
        active["results"][0]["status"] == "blocked"
        and api.called("DELETE", f"/evaluate/workflow-runs/{experiment_id}") == []
    )
    lens.workflows[experiment_id]["status"] = "cancelled"
    done, is_error = await run(
        ctx, "resource_delete", items=[{"kind": "experiment", "id": experiment_id, "name": "ab"}]
    )
    assert not is_error and done["results"][0]["status"] == "deleted"
    assert len(api.called("DELETE", f"/evaluate/workflow-runs/{experiment_id}")) == 1
    assert [item["kind"] for item in done["results"][0]["also_deletes"]].count("configuration_artifact") == 2
    theirs = foreign(lens, status="succeeded")
    refused, _ = await run(ctx, "resource_delete", items=[{"kind": "experiment", "id": theirs, "name": "theirs"}])
    assert (
        refused["results"][0]["error"] == "not_owner"
        and api.called("DELETE", f"/evaluate/workflow-runs/{theirs}") == []
    )
    nothing, is_error = await run(ctx, "resource_delete", items=[])
    assert is_error and nothing["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_list_resources_shows_lens_experiments_with_ownership(api, ctx, lens):
    mine, theirs = running_workflow(lens), foreign(lens)
    own, _ = await run(ctx, "resource_list", kinds=["experiment"])
    assert [item["id"] for item in own["resources"]] == [mine] and own["resources"][0]["ownership"] == "owned"
    everyone, _ = await run(ctx, "resource_list", kinds=["experiment"], owner="all")
    states = {item["id"]: item["ownership"] for item in everyone["resources"]}
    assert states == {mine: "owned", theirs: "shared"}


@pytest.mark.asyncio
async def test_deploying_case_reads_its_deployment_run_and_cluster_is_resolved(api, ctx, lens):
    experiment_id = running_workflow(lens)
    case = lens.workflows[experiment_id]["cases"][0]
    case.update(status="deploying", deployment_run_id="run-9", evaluation_run_id=None)
    lens.workflows[experiment_id]["status"] = "deploying"
    api.on("GET", "/deployments/runs/run-9", {"id": "run-9", "status": "running", "phase": "waiting_for_pods"})
    view, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id)
    assert not is_error and view["active_case"]["phase"] == "waiting_for_pods"
    api.on("GET", "/cluster/clusters", {"items": [{"id": "c1"}]})
    payload, is_error = await run(
        ctx, "experiment_plan", **{key: value for key, value in DESIGN.items() if key != "cluster_id"}
    )
    assert not is_error and payload["cluster_id"] == "c1"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["../runs/ev-1", "a/b", "x?y=1", ".hidden", "ok\n"])
async def test_ids_that_are_not_single_path_segments_never_reach_lens(api, ctx, lens, bad_id):
    mine = running_workflow(lens)
    lens.workflows[mine]["status"] = "cancelled"
    before = len(api.calls)
    payload, is_error = await run(ctx, "resource_delete", items=[{"kind": "experiment", "id": bad_id, "name": "x"}])
    assert is_error and payload["error"]["code"] == "invalid_argument"
    for tool_name, arguments in (
        ("experiment_get", {"experiment_id": bad_id}),
        ("experiment_cancel", {"experiment_id": bad_id}),
        ("experiment_retry", {"experiment_id": bad_id}),
        ("result_compare", {"experiment_ids": [bad_id]}),
        ("result_compare", {"benchmark_run_ids": [bad_id]}),
    ):
        payload, is_error = await run(ctx, tool_name, **arguments)
        assert is_error and payload["error"]["code"] == "invalid_argument", tool_name
    assert len(api.calls) == before
    with pytest.raises(ToolError):
        await evaluation_ops.delete(ctx, "../runs/ev-1")
    assert len(api.calls) == before


@pytest.mark.asyncio
async def test_a_stale_name_refuses_the_whole_experiment_delete(api, ctx, lens):
    mine = running_workflow(lens)
    lens.workflows[mine]["status"] = "cancelled"
    wrong, is_error = await run(ctx, "resource_delete", items=[{"kind": "experiment", "id": mine, "name": "other"}])
    assert is_error and wrong["error"]["code"] == "id_mismatch" and api.called("DELETE", f"{WF}/{mine}") == []
    done, is_error = await run(ctx, "resource_delete", items=[{"kind": "experiment", "id": mine, "name": "mine"}])
    assert not is_error and done["results"][0]["status"] == "deleted"


@pytest.mark.asyncio
async def test_one_unreadable_experiment_does_not_abort_the_batch(api, ctx, lens):
    broken, timing_out, fine = running_workflow(lens), running_workflow(lens), running_workflow(lens)
    lens.workflows[fine]["status"] = "cancelled"
    api.on("GET", f"{WF}/{broken}", LensApiError(503, {"detail": "backend down"}, "GET", "x"))

    def hang(*_):
        raise TimeoutError

    api.on("GET", f"{WF}/{timing_out}", hang)
    items = [
        {"kind": "experiment", "id": experiment_id, "name": "mine"} for experiment_id in (broken, timing_out, fine)
    ]
    payload, is_error = await run(ctx, "resource_delete", items=items)
    assert is_error  # an unreadable experiment cannot have its name checked, so nothing is deleted
    assert api.called("DELETE", f"{WF}/{fine}") == []


@pytest.mark.asyncio
async def test_a_preparing_benchmark_is_not_stale_without_a_heartbeat(api, ctx, lens, monkeypatch):
    experiment_id = running_workflow(lens)
    lens.workflows[experiment_id]["cases"][0]["started_at"] = "2026-05-01T10:01:00+00:00"
    lens.runs["ev-1"] = {
        "id": "ev-1",
        "status": "running",
        "phase": "preparing-runtime",
        "started_at": "2026-05-01T10:01:00+00:00",
        "phase_started_at": "2026-05-01T10:09:30+00:00",
    }
    now = evaluation_view.parse_time("2026-05-01T10:10:30+00:00")
    monkeypatch.setattr(evaluation_view, "now", lambda: now)
    view, is_error = await run(ctx, "experiment_get", experiment_id=experiment_id)
    assert not is_error and view["stale"] == {"is_stale": False}
    assert view["last_progress_at"] == {
        "value": "2026-05-01T10:09:30+00:00",
        "source": "evaluation_run.phase_started_at",
    }
    later = evaluation_view.parse_time("2026-05-01T10:12:00+00:00")
    monkeypatch.setattr(evaluation_view, "now", lambda: later)
    view, _ = await run(ctx, "experiment_get", experiment_id=experiment_id)
    assert view["stale"]["is_stale"]
