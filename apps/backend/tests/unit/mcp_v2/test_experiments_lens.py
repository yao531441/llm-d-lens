"""Experiments on Lens Evaluation workflows. Request bodies are validated with the real Lens models."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from llm_d_lens.configuration.router import RenderRequest, SaveRequest
from llm_d_lens.deploy.contracts import DeployableConfiguration
from llm_d_lens.evaluation.models import EvaluationCreateRequest
from llm_d_lens.evaluation.router import _evaluation_cases, _ordered_evaluation_cases
from llm_d_lens.mcp_v2 import evaluation_ops, evaluation_view
from llm_d_lens.mcp_v2.errors import ToolError
from llm_d_lens.mcp_v2.evaluation_design import ExperimentDesign, evaluation_request, expand, predicted_cases
from llm_d_lens.mcp_v2.guide_fields import GuidePlanFields, normalize_fields
from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.registry import call_tool

from .backend_routes import record_route_problems
from .conftest import FakeApi, plan_routes
from .test_experiments import succeeded_run

WF = "/evaluate/workflow-runs"
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
    first, is_error = await run(ctx, "plan_experiment", **DESIGN)
    assert not is_error, first
    second, _ = await run(ctx, "plan_experiment", **DESIGN)
    assert first["request_digest"] == second["request_digest"] and first["seed"] == second["seed"]
    assert first["cases_total"] == 2 and first["peak_gpus"] == 2 and first["deployments_total"] == 2
    assert first["eta_seconds"]["benchmark"]["upper"] > 0 and first["status"] in ("valid", "valid_with_warnings")
    assert [item["variant_id"] for item in first["plans"]] == ["base", "tp2"]
    assert api.called("POST", "/configurations/save") == [] and api.called("POST", "/evaluate/evaluations") == []
    changed, _ = await run(ctx, "plan_experiment", **{**DESIGN, "scenarios": [SCENARIO_B]})
    assert changed["request_digest"] != first["request_digest"]


@pytest.mark.asyncio
async def test_plan_warns_about_shared_deployment_repetitions_and_counts_cases(api, ctx, lens):
    payload, is_error = await run(ctx, "plan_experiment", **{**DESIGN, "repetitions": {"count": 3}})
    assert not is_error, payload
    assert payload["cases_total"] == 6
    assert "REPEATS_SHARE_DEPLOYMENT" in {warning["code"] for warning in payload["warnings"]}


@pytest.mark.asyncio
async def test_plan_rejects_inapplicable_and_oversized_designs(api, ctx, lens):
    pd_only = {**DESIGN, "variants": [{"id": "base", "plan": {"prefillReplicas": 1}}]}
    payload, is_error = await run(ctx, "plan_experiment", **pd_only)
    assert is_error and payload["error"]["code"] == "field_not_applicable"
    many = {**DESIGN, "scenarios": [{**SCENARIO, "id": f"s{index}"} for index in range(11)]}
    payload, is_error = await run(ctx, "plan_experiment", **{**many, "repetitions": {"count": 2}})
    assert is_error and payload["error"]["code"] == "limit_exceeded"
    assert api.called("POST", "/guide-planning/plan") == []


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
    plan, _ = await run(ctx, "plan_experiment", **DESIGN)
    started, is_error = await run(ctx, "start_experiment", **DESIGN, expected_digest=plan["request_digest"])
    assert not is_error, started
    assert len(api.called("POST", "/configurations/save")) == 2 and lens.saved == 2
    (workflow,) = lens.workflows.values()
    assert started["experiment_id"] == workflow["id"] and started["status"] == "queued"
    assert started["cases_total"] == 2 and started["poll_after_seconds"] and "get_experiment" in started["next"]
    assert [case["id"] for case in started["case_order"]] == [case["id"] for case in workflow["cases"]]
    assert {call[0:2] for call in api.calls if call[0] != "GET" and call[1].startswith("/evaluate")} == {
        ("POST", "/evaluate/evaluations"),
        ("POST", "/evaluate/timing-estimate"),
    }
    assert not any(path.endswith("/cancel") or "/deployments" in path for _, path, _ in api.calls)


@pytest.mark.asyncio
async def test_start_refuses_a_stale_digest_before_saving_anything(api, ctx, lens):
    payload, is_error = await run(ctx, "start_experiment", **DESIGN, expected_digest="sha256:stale")
    assert is_error and payload["error"]["code"] == "plan_stale" and payload["error"]["current_digest"]
    assert api.called("POST", "/configurations/save") == [] and lens.workflows == {}


@pytest.mark.asyncio
async def test_failed_create_removes_the_artifacts_it_saved(api, ctx, lens):
    api.on(
        "POST",
        "/evaluate/evaluations",
        LensApiError(409, {"detail": "Insufficient accelerator capacity for this evaluation"}, "POST", "x"),
    )
    payload, is_error = await run(ctx, "start_experiment", **DESIGN)
    assert is_error and payload["error"]["code"] == "insufficient_capacity"
    assert payload["error"]["leftover_artifacts"] == []
    deleted = [path for method, path, _ in api.calls if method == "DELETE"]
    assert sorted(deleted) == sorted(f"/configurations/artifacts/{uuid(n)}" for n in (1, 2))


@pytest.mark.asyncio
async def test_repeating_a_client_request_id_creates_one_workflow(api, ctx, lens):
    args = {**DESIGN, "client_request_id": "req-1"}
    first, _ = await run(ctx, "start_experiment", **args)
    second, _ = await run(ctx, "start_experiment", **args)
    assert second["experiment_id"] == first["experiment_id"] and len(lens.workflows) == 1
    (workflow,) = lens.workflows.values()
    assert workflow["name"].endswith("[req:req-1]")


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
    view, is_error = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert not is_error, view
    assert view["progress"]["queued"] == 1 and view["progress"]["active"] == 1
    assert view["active_case"]["case_id"] == "guide-1-1" and view["stale"] == {"is_stale": False}
    assert view["last_progress_at"]["value"] == "2026-05-01T10:10:00+00:00" and view["poll_after_seconds"] >= 15
    assert api.called("GET", f"/evaluate/workflow-runs/{experiment_id}/details") == []
    assert "waited_seconds" not in view
    lens.runs["ev-1"]["heartbeat_at"] = "2026-05-01T10:05:00+00:00"
    stale, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
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
    view, is_error = await run(ctx, "get_experiment", experiment_id=experiment_id, wait_seconds=40)
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
    view, _ = await run(ctx, "get_experiment", experiment_id=experiment_id, wait_seconds=10, wait_for="terminal")
    assert view["wait_outcome"] == "timeout" and sum(slept) == 10
    over, is_error = await run(ctx, "get_experiment", experiment_id=experiment_id, wait_seconds=51)
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
    view, is_error = await run(ctx, "get_experiment", experiment_id=experiment_id, include=["comparison", "cases"])
    assert not is_error, view
    assert len(api.called("GET", f"/evaluate/workflow-runs/{experiment_id}/details")) == 1
    assert {item["plan"] for item in view["results"]} == {"base", "tp2"} and len(view["cases"]) == 2
    row = next(r for r in view["comparison"]["rows"] if r["plan"] == "tp2" and r["metric"] == "ttft_ms.p99")
    assert row["delta_vs_baseline_pct"] == 100.0
    compared, is_error = await run(ctx, "compare_results", experiment_ids=[experiment_id], baseline="tp2")
    assert not is_error, compared
    assert compared["comparison"]["baseline"] == "tp2"
    bad, is_error = await run(ctx, "compare_results", experiment_ids=[experiment_id], baseline="nope")
    assert is_error and bad["error"]["field"] == "baseline"


@pytest.mark.asyncio
async def test_unknown_and_legacy_ids_are_routed_by_prefix(api, ctx, lens):
    missing, is_error = await run(ctx, "get_experiment", experiment_id=uuid(4242))
    assert is_error and missing["error"]["code"] == "not_found"
    before = len(api.calls)
    legacy, is_error = await run(ctx, "get_experiment", experiment_id="exp-nope")
    assert is_error and legacy["error"]["code"] == "not_found" and len(api.calls) == before


@pytest.mark.asyncio
async def test_cancel_is_limited_to_the_creator_and_reports_cleanup(api, ctx, lens):
    theirs = foreign(lens)
    refused, is_error = await run(ctx, "cancel_experiment", experiment_id=theirs)
    assert (
        is_error
        and refused["error"]["code"] == "not_owner"
        and api.called("POST", f"/evaluate/workflow-runs/{theirs}/cancel") == []
    )
    unknown = foreign(lens, owner=None)
    refused, _ = await run(ctx, "cancel_experiment", experiment_id=unknown)
    assert refused["error"]["code"] == "owner_unknown"
    mine = running_workflow(lens)
    done, is_error = await run(ctx, "cancel_experiment", experiment_id=mine, reason="wrong model")
    assert not is_error, done
    assert done["status"] == "cancelled" and done["cleanup"]["state"] == "done"
    again, _ = await run(ctx, "cancel_experiment", experiment_id=mine)
    assert again["outcome"] == "already_done" and len(api.called("POST", f"/evaluate/workflow-runs/{mine}/cancel")) == 1


@pytest.mark.asyncio
async def test_cancel_case_and_cleanup_failure(api, ctx, lens):
    mine = running_workflow(lens)
    unknown, is_error = await run(ctx, "cancel_experiment", experiment_id=mine, case_ids=["nope"])
    assert is_error and unknown["error"]["field"] == "case_ids"
    api.on("POST", f"/evaluate/workflow-runs/{mine}/cases/guide-2-1/cancel", lambda *_: {"id": "guide-2-1"})
    single, is_error = await run(ctx, "cancel_experiment", experiment_id=mine, case_ids=["guide-2-1"])
    assert not is_error, single
    assert api.called("POST", f"/evaluate/workflow-runs/{mine}/cancel") == []
    api.on(
        "POST", f"/evaluate/workflow-runs/{mine}/cancel", LensApiError(409, {"detail": "cleanup failed"}, "POST", "x")
    )
    pending, is_error = await run(ctx, "cancel_experiment", experiment_id=mine)
    assert is_error and pending["error"]["code"] == "cleanup_pending" and pending["error"]["retryable"]


@pytest.mark.asyncio
async def test_retry_only_failed_or_cancelled_experiments_of_the_creator(api, ctx, lens):
    running = running_workflow(lens)
    refused, is_error = await run(ctx, "retry_experiment", experiment_id=running)
    assert is_error and refused["error"]["code"] == "conflict_state"
    lens.workflows[running]["status"] = "failed"
    lens.workflows[running]["cases"][0]["status"] = "failed"
    done, is_error = await run(ctx, "retry_experiment", experiment_id=running)
    assert not is_error, done
    assert done["status"] == "queued" and done["requeued_case_ids"] == ["guide-1-1"]
    theirs = foreign(lens, status="failed")
    other, _ = await run(ctx, "retry_experiment", experiment_id=theirs)
    assert other["error"]["code"] == "not_owner"


@pytest.mark.asyncio
async def test_delete_removes_terminal_experiments_and_their_artifacts(api, ctx, lens):
    started, _ = await run(ctx, "start_experiment", **DESIGN)
    experiment_id = started["experiment_id"]
    active, _ = await run(ctx, "delete_resources", items=[{"kind": "experiment", "id": experiment_id}])
    assert (
        active["results"][0]["status"] == "blocked"
        and api.called("DELETE", f"/evaluate/workflow-runs/{experiment_id}") == []
    )
    lens.workflows[experiment_id]["status"] = "cancelled"
    done, is_error = await run(ctx, "delete_resources", items=[{"kind": "experiment", "id": experiment_id}])
    assert not is_error and done["deleted"] == [experiment_id]
    assert len(api.called("DELETE", f"/evaluate/workflow-runs/{experiment_id}")) == 1
    assert len(done["results"][0]["artifacts_deleted"]) == 2
    theirs = foreign(lens, status="succeeded")
    refused, _ = await run(ctx, "delete_resources", items=[{"kind": "experiment", "id": theirs}])
    assert (
        refused["results"][0]["error"] == "not_owner"
        and api.called("DELETE", f"/evaluate/workflow-runs/{theirs}") == []
    )
    nothing, is_error = await run(ctx, "delete_resources")
    assert is_error and nothing["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_list_resources_shows_lens_experiments_with_ownership(api, ctx, lens):
    mine, theirs = running_workflow(lens), foreign(lens)
    own, _ = await run(ctx, "list_resources", kinds=["experiment"])
    assert [item["id"] for item in own["resources"]] == [mine] and own["resources"][0]["ownership"] == "owned"
    everyone, _ = await run(ctx, "list_resources", kinds=["experiment"], owner="all")
    states = {item["id"]: item["ownership"] for item in everyone["resources"]}
    assert states == {mine: "owned", theirs: "shared"}


@pytest.mark.asyncio
async def test_deploying_case_reads_its_deployment_run_and_cluster_is_resolved(api, ctx, lens):
    experiment_id = running_workflow(lens)
    case = lens.workflows[experiment_id]["cases"][0]
    case.update(status="deploying", deployment_run_id="run-9", evaluation_run_id=None)
    lens.workflows[experiment_id]["status"] = "deploying"
    api.on("GET", "/deployments/runs/run-9", {"id": "run-9", "status": "running", "phase": "waiting_for_pods"})
    view, is_error = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert not is_error and view["active_case"]["phase"] == "waiting_for_pods"
    api.on("GET", "/cluster/clusters", {"items": [{"id": "c1"}]})
    payload, is_error = await run(
        ctx, "plan_experiment", **{key: value for key, value in DESIGN.items() if key != "cluster_id"}
    )
    assert not is_error and payload["cluster_id"] == "c1"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["../runs/ev-1", "a/b", "x?y=1", ".hidden", "ok\n"])
async def test_ids_that_are_not_single_path_segments_never_reach_lens(api, ctx, lens, bad_id):
    mine = running_workflow(lens)
    lens.workflows[mine]["status"] = "cancelled"
    before = len(api.calls)
    payload, is_error = await run(ctx, "delete_resources", items=[{"kind": "experiment", "id": bad_id}])
    assert is_error and payload["error"]["code"] == "invalid_argument"
    for tool_name, arguments in (
        ("get_experiment", {"experiment_id": bad_id}),
        ("cancel_experiment", {"experiment_id": bad_id}),
        ("retry_experiment", {"experiment_id": bad_id}),
        ("compare_results", {"experiment_ids": [bad_id]}),
        ("compare_results", {"benchmark_run_ids": [bad_id]}),
    ):
        payload, is_error = await run(ctx, tool_name, **arguments)
        assert is_error and payload["error"]["code"] == "invalid_argument", tool_name
    assert len(api.calls) == before
    direct = await evaluation_ops.delete(ctx, "../runs/ev-1")
    assert direct["status"] == "failed" and direct["error"] == "invalid_argument" and len(api.calls) == before


@pytest.mark.asyncio
async def test_expected_names_apply_to_experiment_items(api, ctx, lens):
    mine = running_workflow(lens)
    lens.workflows[mine]["status"] = "cancelled"
    item = {"kind": "experiment", "id": mine}
    wrong, is_error = await run(ctx, "delete_resources", items=[item], expected_names=["other"])
    assert is_error and wrong["error"]["code"] == "id_mismatch" and api.called("DELETE", f"{WF}/{mine}") == []
    short, is_error = await run(ctx, "delete_resources", items=[item], expected_names=[])
    assert is_error and short["error"]["field"] == "expected_names"
    done, is_error = await run(ctx, "delete_resources", items=[item], expected_names=["mine"])
    assert not is_error and done["deleted"] == [mine]


@pytest.mark.asyncio
async def test_one_unreadable_experiment_does_not_abort_the_batch(api, ctx, lens):
    broken, timing_out, fine = running_workflow(lens), running_workflow(lens), running_workflow(lens)
    lens.workflows[fine]["status"] = "cancelled"
    api.on("GET", f"{WF}/{broken}", LensApiError(503, {"detail": "backend down"}, "GET", "x"))

    def hang(*_):
        raise TimeoutError

    api.on("GET", f"{WF}/{timing_out}", hang)
    items = [{"kind": "experiment", "id": experiment_id} for experiment_id in (broken, timing_out, fine)]
    payload, is_error = await run(ctx, "delete_resources", items=items)
    assert not is_error
    statuses = {result["id"]: result for result in payload["results"]}
    assert statuses[broken]["status"] == "failed" and "backend down" in statuses[broken]["message"]
    assert statuses[timing_out]["status"] == "failed" and statuses[timing_out]["error"] == "timeout"
    assert statuses[fine]["status"] == "deleted" and payload["summary"]["failed"] == 2


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
    view, is_error = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert not is_error and view["stale"] == {"is_stale": False}
    assert view["last_progress_at"] == {
        "value": "2026-05-01T10:09:30+00:00",
        "source": "evaluation_run.phase_started_at",
    }
    later = evaluation_view.parse_time("2026-05-01T10:12:00+00:00")
    monkeypatch.setattr(evaluation_view, "now", lambda: later)
    view, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert view["stale"]["is_stale"]
