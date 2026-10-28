import pytest

from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.lens_api import LensApiError
from llm_d_lens.mcp_v2.registry import all_tools, call_tool

from .conftest import FakeApi, plan_routes

SPEC = {"guide": "optimized-baseline", "model": "Qwen/Qwen3-0.6B"}
SPEC_VARIANT = {"variant_id": "base", "spec": SPEC}


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


def succeeded_run(run_id: str, ttft_p99: float = 800.0, tps: float = 500.0, model: str = SPEC["model"]) -> dict:
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


class Lens:
    """Mutable fake Lens state shared by the routes."""

    def __init__(self, api: FakeApi):
        self.exec_status = "deploying"
        self.runs: dict[str, dict] = {}
        self.created = 0
        api.on("POST", "/evaluate/timing-estimate", {"lower_seconds": 60, "upper_seconds": 120, "timeout_seconds": 600})
        api.on("POST", "/deployments/runs", {"id": "run-1", "status": "queued", "cases": []})
        api.on("GET", "/deployments/runs/run-1", {"id": "run-1", "cases": [{"execution_id": "exec-1"}]})
        api.on("GET", "/deployments/executions/exec-1", lambda *_: {"status": self.exec_status})
        api.on("DELETE", "/deployments/executions/exec-1", {})
        api.on("POST", "/evaluate/runs", self.create_run)
        for run_id in ("ev-1", "ev-2"):
            api.on("GET", f"/evaluate/runs/{run_id}", lambda *_, run_id=run_id: self.runs[run_id])
            api.on("DELETE", f"/evaluate/runs/{run_id}", {})
            api.on("POST", f"/evaluate/runs/{run_id}/cancel", lambda *_, run_id=run_id: self.cancel(run_id))

    def create_run(self, params, body) -> dict:
        self.created += 1
        run_id = f"ev-{self.created}"
        self.runs[run_id] = {"id": run_id, "status": "running"}
        return {"id": run_id, "status": "queued"}

    def cancel(self, run_id: str) -> dict:
        self.runs[run_id] = {"id": run_id, "status": "cancelled"}
        return {"status": "cancelled"}


@pytest.fixture
def lens(api: FakeApi) -> Lens:
    plan_routes(api)
    return Lens(api)


def borrow(ctx: ToolContext, api: FakeApi, execution_id: str = "exec-b", owner: str | None = None) -> str:
    row = ctx.ledger.add_resource(
        kind="deployment",
        owner=owner or ctx.owner,
        cluster_id="c1",
        name="mine",
        lens_execution_id=execution_id,
        meta={"spec": {"cluster_id": "c1", **SPEC}},
    )
    api.on("GET", f"/deployments/executions/{execution_id}", {"status": "ready"})
    return row.id


async def run(ctx: ToolContext, name: str, **arguments):
    payload, is_error = await call_tool(ctx, name, arguments)
    return payload, is_error


async def start_spec(ctx: ToolContext, **extra) -> str:
    payload, is_error = await run(ctx, "start_experiment", cluster_id="c1", variants=[SPEC_VARIANT], **extra)
    assert not is_error, payload
    return payload["experiment_id"]


def test_annotations_follow_the_delta_table():
    tools = {spec.name: spec for spec in all_tools()}
    for name in ("plan_experiment", "get_experiment", "compare_results"):
        assert tools[name].read_only and tools[name].idempotent and not tools[name].destructive
    assert not tools["start_experiment"].read_only and not tools["start_experiment"].destructive
    assert tools["start_experiment"].open_world
    cancel = tools["cancel_experiment"]
    assert cancel.destructive and cancel.idempotent and not cancel.read_only


@pytest.mark.asyncio
async def test_plan_reports_estimates_and_creates_nothing(api, ctx, lens):
    payload, is_error = await run(
        ctx,
        "plan_experiment",
        cluster_id="c1",
        variants=[SPEC_VARIANT, {"variant_id": "tp2", "spec": {**SPEC, "tensor_parallel_size": 2}}],
        rigor="standard",
    )
    assert not is_error, payload
    assert payload["status"] in ("valid", "valid_with_warnings") and payload["plan_id"]
    assert payload["benchmark_runs"] == 2 * 1 * 3
    assert payload["resources"]["peak_gpus"] == 2 and payload["resources"]["gpu_hours_estimate"] > 0
    assert any(w["code"] == "UNEQUAL_GPU_BUDGET" for w in payload["validation"]["warnings"])
    assert api.called("POST", "/deployments/runs") == [] and api.called("POST", "/evaluate/runs") == []


@pytest.mark.asyncio
async def test_plan_over_budget_is_invalid_without_plan_id(api, ctx, lens):
    payload, is_error = await run(ctx, "plan_experiment", cluster_id="c1", variants=[SPEC_VARIANT], max_gpu_hours=0.001)
    assert not is_error and payload["status"] == "invalid" and "plan_id" not in payload
    assert payload["validation"]["errors"][0]["code"] == "MAX_GPU_HOURS_EXCEEDED"


@pytest.mark.asyncio
async def test_borrowed_variants_must_be_owned_and_ready(api, ctx, lens):
    unknown, is_error = await run(ctx, "plan_experiment", variants=[{"variant_id": "x", "deployment_id": "dep-nope"}])
    assert is_error and unknown["error"]["code"] == "owner_unknown"
    other = borrow(ctx, api, "exec-o", owner="someone-else")
    refused, is_error = await run(ctx, "plan_experiment", variants=[{"variant_id": "x", "deployment_id": other}])
    assert is_error and refused["error"]["code"] == "not_owner"
    pending = borrow(ctx, api, "exec-p")
    api.on("GET", "/deployments/executions/exec-p", {"status": "deploying"})
    state, is_error = await run(ctx, "plan_experiment", variants=[{"variant_id": "x", "deployment_id": pending}])
    assert is_error and state["error"]["code"] == "conflict_state"


@pytest.mark.asyncio
async def test_interleaved_order_needs_parallel_schedule(api, ctx, lens):
    payload, is_error = await run(ctx, "plan_experiment", cluster_id="c1", variants=[SPEC_VARIANT], order="interleaved")
    assert is_error and payload["error"]["field"] == "order"


@pytest.mark.asyncio
async def test_temporary_deployment_lifecycle_to_results_and_cleanup(api, ctx, lens):
    experiment_id = await start_spec(ctx)
    (dep,) = ctx.ledger.children(experiment_id)
    assert dep.kind == "deployment" and dep.root_id == experiment_id and dep.state == "active"
    (launch,) = api.called("POST", "/deployments/runs")
    assert len(launch["configurations"]) == 1 and launch["provenance"]["experiment_id"] == experiment_id

    waiting, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert waiting["state"]["execution"] in ("queued", "running") and api.called("POST", "/evaluate/runs") == []

    lens.exec_status = "ready"
    started, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    (benchmark,) = api.called("POST", "/evaluate/runs")
    assert benchmark["deployment_execution_id"] == "exec-1" and benchmark["matrix"]
    assert started["progress"]["runs"]["running"] == 1

    lens.runs["ev-1"] = succeeded_run("ev-1")
    done, _ = await run(ctx, "get_experiment", experiment_id=experiment_id, include=["results", "comparison"])
    assert done["state"] == {
        "execution": "succeeded",
        "measurement": {"state": "complete", "valid_runs": 1, "planned_runs": 1},
        "mechanism": {"state": "not_collected"},
        "cleanup": {"state": "done"},
    }
    (result,) = done["results"]
    assert result["result_ref"] == f"{experiment_id}#base/chat"
    assert result["metrics"]["ttft_ms"]["p99"] == 800.0 and result["metrics"]["success_rate"] == 1.0
    assert result["headline_point"]["concurrency"] == 4
    assert api.called("DELETE", "/deployments/executions/exec-1") == [{"delete_namespace": True}]
    assert ctx.ledger.get(dep.id).state == "deleted" and ctx.ledger.get(experiment_id).state == "completed"


@pytest.mark.asyncio
async def test_failed_benchmark_is_reported_and_deployment_still_deleted(api, ctx, lens):
    experiment_id = await start_spec(ctx)
    lens.exec_status = "ready"
    await run(ctx, "get_experiment", experiment_id=experiment_id)
    lens.runs["ev-1"] = {"id": "ev-1", "status": "failed", "error": "guidellm crashed", "stderr": "Traceback ..."}
    done, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert done["state"]["execution"] == "failed" and done["state"]["cleanup"]["state"] == "done"
    assert done["variants"][0]["error"]["untrusted_text"] == {"stderr": "Traceback ..."}
    assert api.called("DELETE", "/deployments/executions/exec-1")


@pytest.mark.asyncio
async def test_cancel_stops_run_then_deletes_deployment_in_order(api, ctx, lens):
    experiment_id = await start_spec(ctx)
    lens.exec_status = "ready"
    await run(ctx, "get_experiment", experiment_id=experiment_id)
    payload, is_error = await run(ctx, "cancel_experiment", experiment_id=experiment_id)
    assert not is_error, payload
    assert payload["cancelled"] and payload["state"]["execution"] == "cancelled"
    assert payload["state"]["cleanup"]["state"] == "done"
    order = [(m, p) for m, p, _ in api.calls if m in ("POST", "DELETE") and ("cancel" in p or "executions" in p)]
    assert order == [("POST", "/evaluate/runs/ev-1/cancel"), ("DELETE", "/deployments/executions/exec-1")]
    again, is_error = await run(ctx, "cancel_experiment", experiment_id=experiment_id)
    assert not is_error and again["cancelled"] and len(api.called("DELETE", "/deployments/executions/exec-1")) == 1


@pytest.mark.asyncio
async def test_cancel_waits_for_a_run_that_is_still_cancelling(api, ctx, lens, monkeypatch):
    monkeypatch.setattr("llm_d_lens.mcp_v2.experiments_engine.CANCEL_WAIT_SECONDS", 0)
    experiment_id = await start_spec(ctx)
    lens.exec_status = "ready"
    await run(ctx, "get_experiment", experiment_id=experiment_id)
    lens.cancel = lambda run_id: {"status": "cancelling"}
    payload, _ = await run(ctx, "cancel_experiment", experiment_id=experiment_id)
    assert not payload["cancelled"] and payload["state"]["execution"] == "cancelling"
    assert api.called("DELETE", "/deployments/executions/exec-1") == []


@pytest.mark.asyncio
async def test_borrowed_deployment_is_benchmarked_and_never_deleted(api, ctx, lens):
    deployment_id = borrow(ctx, api)
    payload, is_error = await run(
        ctx, "start_experiment", variants=[{"variant_id": "mine", "deployment_id": deployment_id}]
    )
    assert not is_error, payload
    experiment_id = payload["experiment_id"]
    (benchmark,) = api.called("POST", "/evaluate/runs")
    assert benchmark["deployment_execution_id"] == "exec-b"
    lens.runs["ev-1"] = succeeded_run("ev-1")
    done, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert done["state"]["execution"] == "succeeded" and done["state"]["cleanup"]["state"] == "not_needed"
    await run(ctx, "cancel_experiment", experiment_id=experiment_id)
    assert not any(m == "DELETE" and p.startswith("/deployments") for m, p, _ in api.calls)
    assert ctx.ledger.get(deployment_id).state == "active" and ctx.ledger.children(experiment_id) == []


@pytest.mark.asyncio
async def test_failed_deployment_skips_runs_and_reports_failure(api, ctx, lens):
    experiment_id = await start_spec(ctx)
    lens.exec_status = "failed"
    api.on("GET", "/deployments/executions/exec-1", {"status": "failed", "failure": "image pull backoff"})
    done, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert done["state"]["execution"] == "failed" and done["state"]["measurement"]["state"] == "none"
    assert done["variants"][0]["untrusted_text"] == {"failure": "image pull backoff"}
    assert api.called("POST", "/evaluate/runs") == [] and api.called("DELETE", "/deployments/executions/exec-1")


@pytest.mark.asyncio
async def test_deleting_the_experiment_cascades_to_run_records_and_temporary_deployments(api, ctx, lens):
    experiment_id = await start_spec(ctx, keep_deployments=False)
    (dep,) = ctx.ledger.children(experiment_id)
    lens.exec_status = "ready"
    await run(ctx, "get_experiment", experiment_id=experiment_id)
    payload, is_error = await run(ctx, "delete_resources", ids=[experiment_id])
    assert not is_error, payload
    assert payload["deleted"] == [experiment_id]
    assert api.called("DELETE", "/evaluate/runs/ev-1") == [None]
    assert ctx.ledger.get(dep.id).state == "deleted" and ctx.ledger.get(experiment_id).state == "deleted"
    order = [(m, p) for m, p, _ in api.calls if m == "DELETE"]
    assert order.index(("DELETE", "/evaluate/runs/ev-1")) < order.index(("DELETE", "/deployments/executions/exec-1"))


@pytest.mark.asyncio
async def test_failed_cleanup_is_reported_as_residual_and_retried(api, ctx, lens):
    experiment_id = await start_spec(ctx)
    lens.exec_status = "ready"
    await run(ctx, "get_experiment", experiment_id=experiment_id)
    lens.runs["ev-1"] = succeeded_run("ev-1")
    api.on("DELETE", "/deployments/executions/exec-1", LensApiError(500, {"detail": "boom"}, "DELETE", "x"))
    broken, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    cleanup = broken["state"]["cleanup"]
    assert broken["state"]["execution"] == "succeeded" and cleanup["state"] == "failed"
    assert cleanup["residuals"][0]["variant_id"] == "base"
    api.on("DELETE", "/deployments/executions/exec-1", {})
    fixed, _ = await run(ctx, "get_experiment", experiment_id=experiment_id)
    assert fixed["state"]["cleanup"]["state"] == "done"


@pytest.mark.asyncio
async def test_same_plan_id_starts_only_one_experiment(api, ctx, lens):
    plan, _ = await run(ctx, "plan_experiment", cluster_id="c1", variants=[SPEC_VARIANT])
    first, _ = await run(ctx, "start_experiment", plan_id=plan["plan_id"])
    second, _ = await run(ctx, "start_experiment", plan_id=plan["plan_id"])
    assert second["experiment_id"] == first["experiment_id"] and second["already_started"]
    assert len(api.called("POST", "/deployments/runs")) == 1


@pytest.mark.asyncio
async def test_compare_results_reports_deltas_and_mismatches(api, ctx, lens):
    first_dep, second_dep = borrow(ctx, api, "exec-b"), borrow(ctx, api, "exec-c")
    refs = []
    for index, (dep, preset, model) in enumerate(
        [(first_dep, "chat", SPEC["model"]), (second_dep, "latency-sweep", "other/model")], start=1
    ):
        payload, is_error = await run(
            ctx,
            "start_experiment",
            variants=[{"variant_id": "v", "deployment_id": dep}],
            workloads=[{"preset": preset}],
        )
        assert not is_error, payload
        experiment_id = payload["experiment_id"]
        lens.runs[f"ev-{index}"] = succeeded_run(f"ev-{index}", ttft_p99=800.0 * index, tps=500.0 / index, model=model)
        await run(ctx, "get_experiment", experiment_id=experiment_id)
        refs.append(f"{experiment_id}#v/{preset}")
    compared, is_error = await run(ctx, "compare_results", result_refs=refs)
    assert not is_error, compared
    kinds = {issue["kind"] for issue in compared["comparability"]["issues"]}
    assert {"workload_mismatch", "model_mismatch"} <= kinds and not compared["comparability"]["comparable"]
    row = compared["comparison"]["rows"][1]
    assert row["delta"]["ttft_ms.p99"] == 1.0 and row["delta"]["output_tokens_per_s"] == -0.5
    assert compared["comparison"]["winner_by"]["ttft_ms.p99"] == refs[0]
    missing, is_error = await run(ctx, "compare_results", result_refs=[refs[0], "exp-zzzz#v/chat"])
    assert is_error and missing["error"]["code"] == "not_found"
