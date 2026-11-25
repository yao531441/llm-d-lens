"""Tests for Evaluate deployment benchmark support."""

import importlib
from types import SimpleNamespace

import pytest

from llm_d_lens.deploy.contracts import DeployableConfiguration
from llm_d_lens.evaluation.models import EvaluationCreateRequest
from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum

router = importlib.import_module("llm_d_lens.evaluation.router")


@pytest.mark.parametrize("guide", [None, "baseline-vllm", "source-composition", "pd-disaggregation"])
def test_endpoint_benchmark_specification_defaults(guide):
    expected = "guides/pd-disaggregation" if guide == "pd-disaggregation" else "guides/optimized-baseline"
    assert router._guide_benchmark_specification(guide) == expected


def test_source_benchmark_specification_preserves_environment_override(monkeypatch):
    monkeypatch.setattr(router, "_embedded_configuration", lambda _: SimpleNamespace(provider_ref="source-composition"))
    monkeypatch.delenv("LLM_D_BENCH_SPECIFICATION_FILE", raising=False)
    assert router._benchmark_specification({"kind": "candidate"}) == "guides/optimized-baseline"
    monkeypatch.setenv("LLM_D_BENCH_SPECIFICATION_FILE", "custom/spec")
    assert router._benchmark_specification({"kind": "candidate"}) == "custom/spec"


def test_xpumd_metrics_use_compute_engine_utilization():
    from llm_d_lens.hardware.registry import get_profile

    metrics = router._parse_xpumd_metrics(
        "\n".join(
            [
                'hw_gpu_utilization_ratio{pci_bdf="0000:01:00.0",hw_gpu_task="all"} 0.125',
                'hw_gpu_utilization_ratio{pci_bdf="0000:01:00.0",hw_gpu_task="compute-all"} 1',
                'hw_gpu_utilization_ratio{pci_bdf="0000:01:00.0",hw_gpu_task="copy-all"} 0',
            ]
        ),
        get_profile("intel-xpu"),
    )

    assert metrics["average_utilization_ratio"] == 1
    assert metrics["maximum_utilization_ratio"] == 1
    assert metrics["devices"][0]["utilization_ratio"] == 1


@pytest.mark.parametrize(
    ("baseline_type", "provider_ref", "router_profile"),
    [
        ("direct-vllm", "baseline-vllm", None),
        ("router-neutral", "optimized-baseline", "router-neutral"),
        ("optimized-baseline", "optimized-baseline", "optimized-baseline"),
    ],
)
def test_explicit_baseline_types_select_expected_provider(baseline_type, provider_ref, router_profile):
    content = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {"replicaCount": 2, "tensorParallelSize": 1},
        "runtime": {"image": "example/vllm:test", "imageMode": "use-upstream-image"},
    }
    source = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="manifest",
        content=content,
        checksum=configuration_checksum(content),
    )

    baseline = router.baseline_configuration(source, baseline_type, ["workload"], "artifact")

    assert baseline.provider_ref == provider_ref
    assert baseline.content.get("routerProfile") == router_profile
    assert baseline.content["decode"] == content["decode"]
    assert baseline.checksum == configuration_checksum(baseline.content)


def test_comparison_core_parameters_override_generated_baseline():
    content = {
        "model": {"name": "Qwen/Qwen3-8B"},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1, "maxModelLen": 20480},
        "runtime": {"image": "example/vllm:test", "imageMode": "use-upstream-image"},
        "customParameters": [{"target": "decode", "kind": "argument", "name": "block-size", "value": "16"}],
    }
    source = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="manifest",
        content=content,
        checksum=configuration_checksum(content),
    )

    baseline = router.baseline_configuration(
        source,
        "direct-vllm",
        ["workload"],
        "artifact",
        {
            "replicas": 2,
            "tensor_parallel_size": 1,
            "max_model_len": 16384,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.85,
            "block_size": 64,
            "max_num_batched_tokens": 2048,
        },
    )

    assert baseline.content["decode"] == {
        "replicaCount": 2,
        "tensorParallelSize": 1,
        "maxModelLen": 16384,
        "maxNumSeqs": 64,
    }
    arguments = {item["name"]: item["value"] for item in baseline.content["customParameters"]}
    assert arguments == {
        "gpu-memory-utilization": "0.85",
        "block-size": "64",
        "max-num-batched-tokens": "2048",
    }
    assert baseline.checksum == configuration_checksum(baseline.content)


def test_evaluation_case_links_candidate_to_multiple_baselines():
    content = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1},
        "runtime": {"image": "example/vllm:test", "imageMode": "use-upstream-image"},
    }
    configuration = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="manifest",
        content=content,
        checksum=configuration_checksum(content),
    )
    artifact_id = "00000000-0000-4000-8000-000000000001"
    request = EvaluationCreateRequest(
        cluster_session_id="00000000-0000-4000-8000-000000000002",
        benchmark_plans=[
            {
                "id": "candidate",
                "configuration_artifact_id": artifact_id,
                "baseline_types": ["direct-vllm", "router-neutral", "optimized-baseline"],
            }
        ],
    )

    cases = router._evaluation_cases(request, {artifact_id: SimpleNamespace(deployable_configuration=configuration)})

    assert cases[0]["kind"] == "guide"
    assert cases[0]["baseline_case_ids"] == ["baseline-1", "baseline-2", "baseline-3"]
    assert [case["baseline_type"] for case in cases[1:]] == ["direct-vllm", "router-neutral", "optimized-baseline"]


@pytest.mark.parametrize(
    "provider_ref",
    ["optimized-baseline", "pd-disaggregation", "tiered-prefix-cache", "precise-prefix-cache-routing"],
)
def test_suite_scenarios_expand_for_every_supported_guide(provider_ref):
    content = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1},
        "runtime": {"image": "example/vllm:test"},
    }
    if provider_ref == "pd-disaggregation":
        content["prefill"] = {"replicaCount": 1, "tensorParallelSize": 1}
    configuration = DeployableConfiguration(
        type="baseline",
        provider_ref=provider_ref,
        format="manifest",
        content=content,
        checksum=configuration_checksum(content),
    )
    artifact_id = "00000000-0000-4000-8000-000000000001"
    request = EvaluationCreateRequest(
        cluster_session_id="00000000-0000-4000-8000-000000000002",
        benchmark_plans=[
            {
                "id": "candidate",
                "configuration_artifact_id": artifact_id,
                "include_baseline": False,
                "scenarios": [
                    {"id": "smoke", "name": "Smoke", "benchmark": {"matrix": [{"isl": 128, "osl": 64}]}},
                    {"id": "chat", "name": "Chat", "benchmark": {"matrix": [{"isl": 256, "osl": 128}]}},
                ],
            }
        ],
    )

    cases = router._evaluation_cases(request, {artifact_id: SimpleNamespace(deployable_configuration=configuration)})

    assert [case["id"] for case in cases] == ["guide-1-1", "guide-1-2"]
    assert [case["scenario_id"] for case in cases] == ["smoke", "chat"]
    assert all(case["deployment_configuration"]["provider_ref"] == provider_ref for case in cases)


def test_suite_sla_evaluation_uses_requested_percentiles():
    result = router._sla_evaluation(
        {
            "throughput_tps": 1200,
            "success_rate": 99.5,
            "latency_distributions": {
                "ttft": {"p99_ms": 450},
                "tpot": {"p90_ms": 55},
            },
        },
        {
            "ttft_ms": 500,
            "ttft_percentile": "p99",
            "tpot_ms": 50,
            "tpot_percentile": "p90",
            "throughput_min_tps": 1000,
            "success_rate_min_percent": 99,
        },
    )

    assert result["met"] is False
    assert result["checks"]["ttft"]["met"] is True
    assert result["checks"]["tpot"]["met"] is False


def test_suite_scenarios_share_deployment_until_the_last_scenario():
    workflow = {
        "cases": [
            {"id": "guide-1-1", "kind": "guide", "status": "benchmarking", "deployment_group_id": "candidate:guide"},
            {
                "id": "guide-1-2",
                "kind": "guide",
                "status": "queued",
                "deployment_group_id": "candidate:guide",
                "deployment_run_id": None,
            },
            {
                "id": "guide-2-1",
                "kind": "guide",
                "status": "queued",
                "deployment_group_id": "other:guide",
                "deployment_run_id": None,
            },
        ]
    }

    first, second, other = workflow["cases"]
    router._share_suite_deployment(workflow, first, "deployment-1")

    assert second["deployment_run_id"] == "deployment-1"
    assert other["deployment_run_id"] is None
    assert router._has_pending_suite_scenario(workflow, first)
    second["status"] = "succeeded"
    assert not router._has_pending_suite_scenario(workflow, first)


def test_evaluation_capacity_rejects_model_that_cannot_fit_xpu(monkeypatch):
    monkeypatch.setenv("PRISM_ACCELERATOR_MEMORY_GIB", "24")
    content = {
        "model": {"name": "Qwen/Qwen3-32B"},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1},
        "runtime": {"image": "ghcr.io/llm-d/llm-d-xpu:v0.8.0"},
    }
    configuration = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="helm",
        content=content,
        checksum=configuration_checksum(content),
    )

    with pytest.raises(ValueError, match="use TP>=4"):
        router._validate_evaluation_capacity(configuration)

    configuration.content["decode"]["tensorParallelSize"] = 4
    router._validate_evaluation_capacity(configuration)


def test_evaluation_capacity_rejects_more_cards_than_allowlisted(monkeypatch):
    monkeypatch.setenv("PRISM_ACCELERATOR_MEMORY_GIB", "24")
    monkeypatch.setenv("PRISM_GPU_PCI_ALLOWLIST", "gpu-1,gpu-2,gpu-3,gpu-4")
    content = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {"replicaCount": 2, "tensorParallelSize": 4},
        "runtime": {"image": "ghcr.io/llm-d/llm-d-xpu:v0.8.0"},
    }
    configuration = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="helm",
        content=content,
        checksum=configuration_checksum(content),
    )

    with pytest.raises(ValueError, match="configuration requires 8 XPU cards, but only 4 are available to Prism"):
        router._validate_evaluation_capacity(configuration)


@pytest.mark.parametrize("preserve_deployment", [False, True])
def test_failed_benchmark_always_cleans_owned_deployment(preserve_deployment):
    assert router._should_cleanup_deployment(
        owns_deployment=True,
        benchmark_succeeded=False,
        case={"preserve_deployment": preserve_deployment},
    )


def test_successful_benchmark_keeps_deployment_only_when_requested():
    assert router._should_cleanup_deployment(
        owns_deployment=True, benchmark_succeeded=True, case={"preserve_deployment": False}
    )
    assert not router._should_cleanup_deployment(
        owns_deployment=True, benchmark_succeeded=True, case={"preserve_deployment": True}
    )


def test_evaluation_does_not_clean_external_deployment():
    assert not router._should_cleanup_deployment(
        owns_deployment=False, benchmark_succeeded=False, case={"preserve_deployment": False}
    )


@pytest.mark.asyncio
async def test_delete_terminal_endpoint_benchmark_removes_record_and_results(monkeypatch, tmp_path):
    run_id = "00000000-0000-4000-8000-000000000001"
    run = {"kind": "benchmark", "id": run_id, "status": "failed"}
    results = tmp_path / "results"
    output = results / run_id
    output.mkdir(parents=True)
    (output / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(router, "_results_root", results)
    router._save(run)

    await router.delete_benchmark_run(run_id)

    assert router._get("benchmark", run_id) is None
    assert not output.exists()


@pytest.mark.asyncio
async def test_delete_active_endpoint_benchmark_is_rejected(monkeypatch):
    run_id = "00000000-0000-4000-8000-000000000002"
    monkeypatch.setattr(router, "_get", lambda _kind, _record_id: {"id": run_id, "status": "running"})

    with pytest.raises(router.HTTPException, match="must be cancelled") as raised:
        await router.delete_benchmark_run(run_id)

    assert raised.value.status_code == 409


@pytest.mark.asyncio
async def test_delete_workflow_deletes_owned_deployments_and_records(monkeypatch, tmp_path):
    workflow_id = "00000000-0000-4000-8000-000000000021"
    workflow = {
        "kind": "workflow",
        "id": workflow_id,
        "status": "succeeded",
        "deployment_run_id": "legacy-deployment",
        "cases": [
            {
                "id": "00000000-0000-4000-8000-000000000031",
                "status": "succeeded",
                "deployment_run_id": "deployment-1",
                "evaluation_run_id": "00000000-0000-4000-8000-000000000011",
            },
            {
                "id": "00000000-0000-4000-8000-000000000032",
                "status": "succeeded",
                "deployment_run_id": "deployment-1",
                "evaluation_run_id": "00000000-0000-4000-8000-000000000012",
            },
        ],
    }
    deleted_deployments = []
    monkeypatch.setattr(router, "_results_root", tmp_path / "results")
    monkeypatch.setattr(router, "_records", lambda kind: [])

    class Worker:
        def __init__(self, run_id):
            self.run_id = run_id

        async def delete_run(self, run_id):
            assert run_id == self.run_id
            deleted_deployments.append(run_id)

    monkeypatch.setattr(router, "_store", SimpleNamespace(get_run=lambda _run_id: object()))
    monkeypatch.setattr(
        router,
        "deployment_run_manager",
        SimpleNamespace(worker_for_run=lambda run_id: Worker(run_id)),
    )
    router._save(workflow)
    router._save({"kind": "benchmark", "id": "00000000-0000-4000-8000-000000000011", "status": "failed"})
    router._save({"kind": "benchmark", "id": "00000000-0000-4000-8000-000000000012", "status": "failed"})

    await router.delete_workflow_run(workflow_id)

    assert set(deleted_deployments) == {"legacy-deployment", "deployment-1"}
    assert router._get("workflow", workflow_id) is None
    assert router._get("benchmark", "00000000-0000-4000-8000-000000000011") is None
    assert router._get("benchmark", "00000000-0000-4000-8000-000000000012") is None


@pytest.mark.asyncio
async def test_delete_workflow_keeps_records_when_deployment_cleanup_fails(monkeypatch, tmp_path):
    workflow_id = "00000000-0000-4000-8000-000000000022"
    workflow = {
        "kind": "workflow",
        "id": workflow_id,
        "status": "failed",
        "cases": [
            {
                "id": "00000000-0000-4000-8000-000000000033",
                "status": "failed",
                "deployment_run_id": "deployment-1",
                "evaluation_run_id": "00000000-0000-4000-8000-000000000011",
            }
        ],
    }

    class Worker:
        async def delete_run(self, _run_id):
            raise ValueError("reconnect the deployment cluster")

    router._save(workflow)
    monkeypatch.setattr(router, "_store", SimpleNamespace(get_run=lambda _run_id: object()))
    monkeypatch.setattr(router, "deployment_run_manager", SimpleNamespace(worker_for_run=lambda _run_id: Worker()))

    with pytest.raises(router.HTTPException, match="reconnect the deployment cluster") as raised:
        await router.delete_workflow_run(workflow_id)

    assert raised.value.status_code == 409
    saved = router._get("workflow", workflow_id)
    assert saved is not None
    assert saved["id"] == workflow["id"]
    assert saved["status"] == workflow["status"]
    assert [case["id"] for case in saved["cases"]] == [case["id"] for case in workflow["cases"]]
    assert [case["status"] for case in saved["cases"]] == [case["status"] for case in workflow["cases"]]
    assert [case["deployment_run_id"] for case in saved["cases"]] == [
        case["deployment_run_id"] for case in workflow["cases"]
    ]


@pytest.mark.asyncio
async def test_benchmark_runtime_requires_cluster_source(monkeypatch, tmp_path):
    executable = tmp_path / ".venv" / "bin" / "llmdbenchmark"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    monkeypatch.setenv("LLM_D_BENCHMARK_EXECUTABLE", str(executable))
    monkeypatch.setenv("LLM_D_BENCHMARK_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="Software Versions"):
        await router._prepare_benchmark_runtime()


@pytest.mark.asyncio
async def test_benchmark_runtime_uses_cluster_checkout(tmp_path):
    executable = tmp_path / ".venv" / "bin" / "llmdbenchmark"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    template = tmp_path / "config/templates/jinja/20_harness_pod.yaml.j2"
    template.parent.mkdir(parents=True)
    template.write_text("{{ harness_command }}\n", encoding="utf-8")
    assert await router._prepare_benchmark_runtime({"resolved_repository": str(tmp_path)}) == (
        str(executable),
        tmp_path,
    )


@pytest.mark.asyncio
async def test_workflow_persists_deployment_failure_stage_before_clearing_active_case(monkeypatch):
    import copy

    workflow = {
        "kind": "workflow",
        "id": "stage-test",
        "status": "queued",
        "failed_stage": "benchmarking",
        "cases": [
            {"id": "case-1", "kind": "guide", "status": "queued", "failed_stage": "benchmarking"},
        ],
    }
    saved = []
    monkeypatch.setattr(router, "_get", lambda *_: workflow)
    monkeypatch.setattr(router, "_save", lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(router, "_ordered_evaluation_cases", lambda cases: cases)

    def invalid_configuration(*_):
        raise ValueError("deployment configuration rejected")

    monkeypatch.setattr(router, "_case_configuration", invalid_configuration)
    await router._execute_evaluation(workflow["id"])

    final = saved[-1]
    assert final["status"] == "failed"
    assert final["active_case_id"] is None
    assert final["failed_stage"] == "deploying"
    assert final["cases"][0]["failed_stage"] == "deploying"
    assert final["cases"][0]["error"] == "deployment configuration rejected"
    assert "failed_stage" not in saved[0]
    assert "failed_stage" not in saved[0]["cases"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_count", [1, 2])
@pytest.mark.parametrize("cleanup_error", [None, RuntimeError("cleanup unavailable")])
@pytest.mark.parametrize("shared_pods", [False, True])
async def test_workflow_completes_before_final_deployment_cleanup(monkeypatch, case_count, cleanup_error, shared_pods):
    import asyncio
    import copy

    workflow = {
        "kind": "workflow",
        "id": "completion-test",
        "status": "queued",
        "cases": [
            {"id": f"case-{index}", "kind": "guide", "status": "queued", "deployment_run_id": f"deployment-{index}"}
            for index in range(case_count)
        ],
    }
    saved = []
    if shared_pods:
        workflow["cases"].append(
            {
                "id": "shared-baseline",
                "kind": "baseline",
                "status": "queued",
                "baseline_type": "kubernetes-service",
                "dependent_guide_case_id": f"case-{case_count - 1}",
            }
        )
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = []
    ready_case = SimpleNamespace(
        id="ready-case",
        status=SimpleNamespace(value="ready"),
        execution_id="execution",
        attempt=1,
        provider_ref="optimized-baseline",
    )
    execution = SimpleNamespace(
        status=router.DeploymentStatus.READY,
        namespace="test",
        endpoint=SimpleNamespace(url="http://routed", baseline_url="http://direct"),
    )
    monkeypatch.setattr(router, "_get", lambda *_: workflow)
    monkeypatch.setattr(router, "_save", lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(router, "_ordered_evaluation_cases", lambda cases: cases)
    monkeypatch.setattr(router, "_benchmark_specification", lambda *_: None)
    monkeypatch.setattr(router, "case_benchmark_request", lambda *_, **__: None)
    monkeypatch.setattr(router, "_comparison_report", lambda *_: {"ready": True})
    monkeypatch.setattr(router, "_has_pending_suite_scenario", lambda *_: False)
    monkeypatch.setattr(
        router,
        "_store",
        SimpleNamespace(
            get_run=lambda run_id: SimpleNamespace(id=run_id, cases=[ready_case]),
            get_execution=lambda *_: execution,
        ),
    )

    async def benchmark(_workflow, case, *_args, **_kwargs):
        case.update(status="succeeded", finished_at=router._now())

    class Worker:
        async def delete_case_execution(self, run_id, _case_id):
            cleaned.append(run_id)
            if len(cleaned) < case_count:
                assert saved[-1]["status"] == "running"
                return
            cleanup_started.set()
            await release_cleanup.wait()
            if cleanup_error:
                raise cleanup_error

    monkeypatch.setattr(router, "execute_case_benchmark", benchmark)
    monkeypatch.setattr(router, "deployment_run_manager", SimpleNamespace(worker_for_run=lambda *_: Worker()))
    task = asyncio.create_task(router._execute_evaluation(workflow["id"]))
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert not task.done()
        completed = saved[-1]
        assert completed["status"] == "succeeded"
        assert completed["active_case_id"] is None
        assert completed["report"] == {"ready": True}
        assert completed["finished_at"]
    finally:
        release_cleanup.set()
        await asyncio.wait_for(task, timeout=2)
    assert saved[-1]["status"] == "succeeded"
    assert saved[-1]["finished_at"] == completed["finished_at"]
    assert saved[-1]["cases"][case_count - 1].get("cleanup_error") == (str(cleanup_error) if cleanup_error else None)


@pytest.mark.asyncio
async def test_delete_completed_workflow_waits_for_background_cleanup(monkeypatch):
    import asyncio

    workflow_id = "00000000-0000-4000-8000-000000000023"
    workflow = {"kind": "workflow", "id": workflow_id, "status": "succeeded", "cases": []}
    release_cleanup = asyncio.Event()
    delete_checked = asyncio.Event()

    async def cleanup():
        await release_cleanup.wait()
        router._save(workflow)

    original_get = router._get

    def get_record(kind, record_id):
        record = original_get(kind, record_id)
        delete_checked.set()
        return record

    router._save(workflow)
    monkeypatch.setattr(router, "_get", get_record)
    cleanup_task = asyncio.create_task(cleanup())
    monkeypatch.setitem(router._workflow_tasks, workflow_id, cleanup_task)
    deletion = asyncio.create_task(router.delete_workflow_run(workflow_id))
    try:
        await asyncio.wait_for(delete_checked.wait(), timeout=2)
        assert not deletion.done()
        assert original_get("workflow", workflow_id) is not None
    finally:
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(cleanup_task, deletion), timeout=2)
    assert original_get("workflow", workflow_id) is None


def test_each_configuration_can_select_only_ablations_and_keeps_its_own_comparisons():
    content = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {"replicaCount": 2, "tensorParallelSize": 1},
        "runtime": {"image": "vllm:xpu"},
    }
    config = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="helm",
        content=content,
        checksum=configuration_checksum(content),
    )
    a = "00000000-0000-4000-8000-000000000001"
    b = "00000000-0000-4000-8000-000000000002"
    request = EvaluationCreateRequest(
        cluster_session_id=a,
        benchmark_plans=[
            {
                "id": "a",
                "configuration_artifact_id": a,
                "include_configuration": False,
                "baseline_types": ["load-only", "affinity-only"],
            },
            {"id": "b", "configuration_artifact_id": b, "include_configuration": True, "baseline_types": ["load-only"]},
        ],
    )
    cases = router._evaluation_cases(
        request,
        {a: SimpleNamespace(deployable_configuration=config), b: SimpleNamespace(deployable_configuration=config)},
    )
    assert len(cases) == 4
    assert [(c["kind"], c["benchmark_plan_id"]) for c in cases] == [
        ("guide", "b"),
        ("baseline", "a"),
        ("baseline", "a"),
        ("baseline", "b"),
    ]
    assert cases[0]["baseline_case_ids"] == [cases[-1]["id"]]


def test_router_ablations_preserve_published_source_and_cluster_for_retry():
    content = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {"replicaCount": 2, "tensorParallelSize": 1},
        "runtime": {"image": "vllm:xpu"},
        "officialGuide": {"renderedManifest": "saved-manifest", "deploymentBundle": {"saved": True}},
        "guideSettings": {"routerValues": "saved-router"},
    }
    source = DeployableConfiguration(
        type="baseline",
        provider_ref="optimized-baseline",
        format="manifest",
        content=content,
        checksum=configuration_checksum(content),
        provenance={"cluster_ref": {"id": "original-cluster"}},
    )
    for kind in ["router-neutral", "affinity-only", "load-only"]:
        baseline = router.baseline_configuration(source, kind, ["plan-a"], "artifact-a")
        assert baseline.content["officialGuide"] == content["officialGuide"]
        assert baseline.content["guideSettings"] == content["guideSettings"]
        assert baseline.provenance["cluster_ref"]["id"] == "original-cluster"
        assert baseline.content["routerProfile"] == kind
        assert "routerProfile" not in source.content
    direct = router.baseline_configuration(source, "direct-vllm", ["plan-a"], "artifact-a")
    assert direct.provenance["cluster_ref"]["id"] == "original-cluster"
    assert "officialGuide" not in direct.content


def test_a_case_that_failed_by_exception_does_not_block_its_own_cleanup(monkeypatch):
    from llm_d_lens.deploy import usage

    workflow = {
        "id": "w1",
        "status": "running",
        "cases": [{"id": "c1", "status": "benchmarking", "execution_id": "e1"}],
    }
    monkeypatch.setattr(router, "_records", lambda kind: [workflow] if kind == "workflow" else [])
    monkeypatch.setattr(usage, "_probes", {"evaluate": router._deployment_usage_reason})
    # The failure status is only written after the cleanup in `finally` has run.
    with pytest.raises(usage.DeploymentInUseError):
        usage.require_unused("e1")
    with router._deployment_cleanup_guard("e1"):
        usage.require_unused("e1")
    with pytest.raises(usage.DeploymentInUseError):
        usage.require_unused("e1")
