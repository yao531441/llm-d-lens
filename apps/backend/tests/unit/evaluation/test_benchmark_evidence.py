"""Evaluate evidence must be measured, attributable, immutable and replayable."""

import asyncio
import importlib
import io
import json
import sys
import tarfile
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import zstandard

from llm_d_lens.agentic import benchmark_context
from llm_d_lens.db.dao.agentic_benchmark_record import BenchmarkRecordDao, BenchmarkRecordDaoError
from llm_d_lens.db.dao.evaluate_run import EvaluateRunDao
from llm_d_lens.db.evaluate_persistence_models import EvaluateRunRecord
from llm_d_lens.evaluation import benchmark_evidence as evidence
from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum


@pytest.fixture
def run():
    content = {
        "model": {"name": "org/model"},
        "decode": {"tensorParallelSize": 1, "replicaCount": 1},
    }
    metrics = {
        "throughput_tps": 40.0,
        "throughput_rps": 2.0,
        "success_count": 9,
        "failure_count": 1,
        "request_count": 10,
        "benchmark_time_seconds": 5.0,
        "latency_distributions": {
            "ttft": {"p50_ms": 10.0, "p95_ms": 100.0},
            "tpot": {"p50_ms": 1.0, "p95_ms": 5.0},
            "request_latency": {"p95_ms": 200.0},
        },
        "token_distributions": {"input": {"mean": 100.0, "p95": 120.0}, "output": {"mean": 20.0}},
    }
    return {
        "id": str(uuid4()),
        "kind": "benchmark",
        "status": "succeeded",
        "harness": "inference-perf",
        "parallelism": 1,
        "cluster_id": "cluster1",
        "namespace": "deployment-test",
        "benchmark_started_at": "2026-10-20T12:01:00Z",
        "finished_at": "2026-10-20T12:02:00Z",
        "deployment_configuration": {
            "type": "baseline",
            "format": "helm",
            "provider_ref": "baseline-vllm",
            "checksum": configuration_checksum(content),
            "content": content,
        },
        "matrix": [{"isl": 100, "osl": 20}],
        "matrix_results": [
            {
                "status": "succeeded",
                "isl": 100,
                "osl": 20,
                "stage_metrics": [{"concurrency": 2, "num_requests": 10, "metrics": metrics}],
            }
        ],
        "benchmark_context": {
            "schema_version": 1,
            "source": "serving-container metadata via Kubernetes API",
            "captured_at": "2026-10-20T12:00:00Z",
            "verified_at": "2026-10-20T12:03:00Z",
            "configuration_checksum": configuration_checksum(content),
            "model": {
                "model_id": "org/model",
                "family": "test",
                "architecture": "moe",
                "parameter_count_b": 8.0,
                "active_parameter_count_b": 2.0,
                "quantization": "none",
                "tokenizer_id": "org/model",
                "revision": "pinned-revision",
            },
            "hardware": {"accelerator": "Actual device", "vram_per_gpu_gib": 80.0, "gpu_count": 1},
            "runtime": {"backend": "vllm", "backend_version": "0.11.0", "accelerator_runtime": "torch.xpu:torch-2.8"},
        },
    }


def sample_metrics(run):
    return run["matrix_results"][0]["stage_metrics"][0]["metrics"]


def test_source_evidence_derives_topology_from_reviewed_manifest(run):
    from tests.unit.agentic.test_source_plan import bundle

    content = run["deployment_configuration"]["content"]
    content.pop("decode")
    content["officialGuide"] = {
        "renderedManifest": bundle()["files"][0]["content"],
        "source": {"files": ["guides/optimized-baseline/fixture"]},
    }
    run["deployment_configuration"].update(provider_ref="source-composition", checksum=configuration_checksum(content))
    _, candidate = evidence.configuration_snapshot(run)
    assert candidate.decode_tp == 1
    assert candidate.decode_replicas == 1
    assert candidate.prefill_tp is None
    content["officialGuide"]["renderedManifest"] = ""
    run["deployment_configuration"]["checksum"] = configuration_checksum(content)
    with pytest.raises(ValueError, match="saved rendered manifest"):
        evidence.configuration_snapshot(run)


@pytest.mark.parametrize("recipe", ["optimized-baseline", "precise-prefix-cache-routing", "tiered-prefix-cache"])
@pytest.mark.parametrize("mounted,referenced", [(False, True), (True, False), (True, True)])
def test_source_evidence_keeps_recipe_and_requires_referenced_router_config(run, recipe, mounted, referenced):
    import yaml

    from llm_d_lens.agentic.source_plan import bundle_documents
    from llm_d_lens.deploy.source_composition import dump
    from tests.unit.agentic.test_source_plan import bundle

    parsed = bundle_documents(bundle())
    pod = parsed[1]["spec"]["template"]["spec"]
    if mounted:
        pod["volumes"] = [{"name": "router", "configMap": {"name": "router"}}]
    config = {
        "kind": "EndpointPickerConfig",
        "plugins": [{"type": "prefix-cache-affinity-filter"}],
        "schedulingProfiles": [{"plugins": [{"pluginRef": "prefix-cache-affinity-filter"}] if referenced else []}],
    }
    parsed.append(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "router"},
            "data": {"policy.yaml": yaml.safe_dump(config)},
        }
    )
    content = run["deployment_configuration"]["content"]
    content["officialGuide"] = {"renderedManifest": dump(parsed), "source": {"files": [f"guides/{recipe}/fixture"]}}
    run["deployment_configuration"].update(provider_ref="source-composition", checksum=configuration_checksum(content))
    _, candidate = evidence.configuration_snapshot(run)
    assert candidate.topology == recipe
    assert f"prefix-routing:{'on' if mounted and referenced else 'off'}" in candidate.optimizations
    content["officialGuide"]["source"]["files"] = []
    run["deployment_configuration"]["checksum"] = configuration_checksum(content)
    with pytest.raises(ValueError, match="unambiguous source recipe"):
        evidence.configuration_snapshot(run)


def test_uses_actual_p95_fraction_and_original_time_not_medians_or_current_facts(run):
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    (record,) = records
    assert record.benchmark_id == f"evaluate:{run['id']}:matrix:0:stage:0"
    assert record.metrics.ttft_p95_ms == 100
    assert record.metrics.tpot_p95_ms == 5
    assert record.metrics.error_rate == 0.1
    assert record.timestamp.isoformat() == "2026-10-20T12:02:00+00:00"
    assert record.model.architecture == "moe"
    assert record.model.active_parameter_count_b == 2
    assert record.hardware.vram_per_gpu_gib == 80
    assert record.workload.mean_input_tokens == 100
    assert record.workload.prefill_fraction == pytest.approx(100 / 120)


@pytest.mark.parametrize("status", ["queued", "running", "cancelled", "cancelling"])
def test_nonmeasured_terminal_or_inflight_runs_are_not_promoted(run, status):
    run["status"] = status
    assert evidence.normalize_evaluate_run(run)[0] == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["benchmark_context"].pop("verified_at"),
        lambda r: r["benchmark_context"]["hardware"].update(vram_per_gpu_gib=float("inf")),
        lambda r: r["benchmark_context"]["hardware"].update(gpu_count=2),
        lambda r: r["benchmark_context"]["model"].update(model_id="other/model"),
        lambda r: r["deployment_configuration"]["content"]["decode"].update(replicaCount=2),
        lambda r: r.update(finished_at="2026-10-20T12:02:00"),
        lambda r: r["benchmark_context"].update(captured_at="2026-10-20T12:02:00Z"),
        lambda r: r.update(use_baseline_endpoint=True),
        lambda r: r.update(parallelism=2),
        lambda r: r.update(harness="unsupported"),
        lambda r: sample_metrics(r).update(throughput_tps=float("nan")),
        lambda r: sample_metrics(r).update(request_count=99),
        lambda r: sample_metrics(r).pop("failure_count"),
        lambda r: sample_metrics(r)["token_distributions"]["input"].pop("p95"),
        lambda r: sample_metrics(r).update(throughput_rps=0),
        lambda r: r["matrix_results"][0]["stage_metrics"][0].update(concurrency=True),
    ],
)
def test_missing_invalid_or_unattributable_evidence_is_rejected_with_reason(run, mutate):
    mutate(run)
    records, issues = evidence.normalize_evaluate_run(run)
    assert records == []
    assert issues and issues[0]["reason"]


def test_no_latency_is_fabricated_when_only_measured_throughput_exists(run):
    sample_metrics(run).pop("latency_distributions")
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    assert records[0].metrics.ttft_p95_ms is None
    assert records[0].metrics.throughput_tokens_per_s == 40
    sample_metrics(run).pop("throughput_tps")
    assert not evidence.normalize_evaluate_run(run)[0]


def test_stages_are_separate_and_invalid_sibling_does_not_poison_valid_measurement(run):
    other = deepcopy(run["matrix_results"][0]["stage_metrics"][0])
    other["metrics"]["throughput_tps"] = -1
    run["matrix_results"][0]["stage_metrics"].append(other)
    records, issues = evidence.normalize_evaluate_run(run)
    assert len(records) == 1
    assert issues[0]["sample"] == ":matrix:0:stage:1"


def test_failed_measured_sample_preserves_failure_but_setup_failure_has_no_record(run):
    run.update(status="failed", error="Measured run failed")
    run["matrix_results"][0]["status"] = "failed"
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    assert records[0].outcome == "failed"
    assert records[0].failure_reason == "Measured run failed"
    run.pop("benchmark_started_at")
    assert not evidence.normalize_evaluate_run(run)[0]


def test_terminal_save_ingests_once_and_survives_evaluate_deletion(run, monkeypatch):
    router = importlib.import_module("llm_d_lens.evaluation.router")
    monkeypatch.setattr(router, "_register_evaluation_artifacts", lambda r: None)
    router._save(run)
    dao = BenchmarkRecordDao()
    (record,) = dao.list()
    assert run["benchmark_evidence"]["status"] == "stored"
    assert run["benchmark_evidence"]["inserted_ids"] == [record.benchmark_id]
    original = record.model_dump()
    router._save(run)
    assert not run["benchmark_evidence"]["inserted_ids"]
    assert len(dao.list()) == 1
    assert dao.list()[0].model_dump() == original
    EvaluateRunDao().delete(run["id"])
    assert dao.get(record.benchmark_id).model_dump() == original


def test_backfill_cli_defaults_to_preview_then_idempotently_applies_only_eligible_runs(run, monkeypatch, capsys):
    dao = EvaluateRunDao()
    dao.save(EvaluateRunRecord.model_validate(run))
    incomplete = deepcopy(run)
    incomplete["id"] = str(uuid4())
    incomplete.pop("benchmark_started_at")
    dao.save(EvaluateRunRecord.model_validate(incomplete))
    monkeypatch.setattr(sys, "argv", ["backfill"])
    evidence.main()
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(reports) == 2
    assert all(not report["apply"] for report in reports)
    assert not BenchmarkRecordDao().list()
    assert "benchmark_evidence" not in dao.get(run["id"]).model_dump()
    monkeypatch.setattr(sys, "argv", ["backfill", "--apply"])
    evidence.main()
    assert len(BenchmarkRecordDao().list()) == 1
    assert dao.get(incomplete["id"]).model_dump()["benchmark_evidence"]["status"] == "skipped"
    evidence.main()
    assert len(BenchmarkRecordDao().list()) == 1


def test_evidence_storage_failure_is_reported_without_losing_evaluate_result(run, monkeypatch):
    router = importlib.import_module("llm_d_lens.evaluation.router")
    monkeypatch.setattr(router, "_register_evaluation_artifacts", lambda r: None)

    def unavailable(self, records):
        raise BenchmarkRecordDaoError("Database unavailable")

    monkeypatch.setattr(BenchmarkRecordDao, "save_many", unavailable)
    router._save(run)
    saved = EvaluateRunDao().get(run["id"]).model_dump()
    assert saved["status"] == "succeeded"
    assert saved["benchmark_evidence"]["status"] == "error"
    assert not BenchmarkRecordDao().list()


def test_shared_prefix_uses_measured_request_overlap_and_explicit_stage_ids(run, tmp_path):
    run.pop("matrix")
    run.pop("matrix_results")
    run["shared_prefix"] = {"system_prompt_len": 80, "question_len": 20}
    metrics = {
        "throughput_tps": 40,
        "throughput_rps": 1,
        "request_count": 2,
        "success_count": 2,
        "failure_count": 0,
        "benchmark_time_seconds": 2,
        "token_distributions": {"input": {"mean": 100, "p95": 100}, "output": {"mean": 20}},
        "summary_path": str(tmp_path / "stage_0_lifecycle_metrics.json"),
    }
    run["rate_stage_results"] = [{"rate": 3, "duration": 2, "metrics": metrics}]
    path = tmp_path / "per_request_lifecycle_metrics.json"
    rows = [
        {"stage_id": 0, "start_time": 0, "end_time": 1, "error": None},
        {"stage_id": 0, "start_time": 0, "end_time": 2, "error": None},
        {"stage_id": -1, "start_time": 0, "end_time": 100, "error": None},
    ]
    path.write_text(json.dumps(rows))
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    assert records[0].workload.concurrency == 1.5
    assert records[0].workload.request_rate == 1
    assert records[0].workload.shared_prefix_ratio == 0.8
    rows[0].pop("stage_id")
    path.write_text(json.dumps(rows))
    assert not evidence.normalize_evaluate_run(run)[0]


def test_standalone_random_workload_is_supported_but_mixed_stages_are_not(run):
    metrics = sample_metrics(run)
    run.pop("matrix")
    run.pop("matrix_results")
    run["metrics"] = metrics
    run["workload_yaml"] = "data:\n  type: random\nload:\n  stages:\n    - concurrency_level: 3\n"
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    assert records[0].workload.concurrency == 3
    run["workload_yaml"] += "    - concurrency_level: 4\n"
    assert not evidence.normalize_evaluate_run(run)[0]


@pytest.mark.parametrize("probe_exit", [0, 1])
@pytest.mark.parametrize("model_source", ["org/model", "other/model"])
def test_metadata_is_collected_from_serving_container_and_restart_invalidates_it(
    run, monkeypatch, probe_exit, model_source
):
    metadata = deepcopy(run.pop("benchmark_context"))
    probe = {key: metadata[key] for key in ("model", "hardware", "runtime")}
    pod = {
        "metadata": {"name": "model-0", "uid": "pod-uid"},
        "spec": {
            "containers": [
                {
                    "name": "modelserver",
                    "command": ["vllm", "serve"],
                    "args": [model_source, "--revision", "pinned-revision", "--tensor-parallel-size=1"],
                    "resources": {"limits": {"test.example/accelerator": "1"}},
                }
            ]
        },
        "status": {
            "containerStatuses": [
                {
                    "name": "modelserver",
                    "ready": True,
                    "imageID": "sha256:actual-image",
                    "restartCount": 0,
                }
            ]
        },
    }
    calls = []

    async def execute(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0 if "get" in argv else probe_exit,
            stdout=json.dumps({"items": [pod]} if "get" in argv else probe),
        )

    monkeypatch.setattr(benchmark_context, "scoped_runner", lambda cluster: SimpleNamespace(run=execute))
    monkeypatch.setattr(benchmark_context, "resolve_by_resource", lambda key: True)
    monkeypatch.setattr(benchmark_context, "enrich_model_metadata", AsyncMock())
    asyncio.run(evidence.capture_benchmark_context(run))
    if model_source != "org/model":
        assert "benchmark_context" not in run
        assert "identity does not match" in run["benchmark_context_error"]
        assert not any("exec" in argv for argv in calls)
        return
    if not probe_exit:
        assert run["benchmark_context"]["model"] == probe["model"]
        assert run["benchmark_context"]["hardware"] == probe["hardware"]
    else:
        assert run["benchmark_context"]["hardware"]["gpu_count"] is None
        assert run["benchmark_context"]["warnings"] == ["metadata probe exit 1"]
    assert any("exec" in argv for argv in calls)
    asyncio.run(evidence.verify_benchmark_context(run))
    assert run["benchmark_context"]["verified_at"]
    pod["status"]["containerStatuses"][0]["restartCount"] = 1
    asyncio.run(evidence.verify_benchmark_context(run))
    assert "verified_at" not in run["benchmark_context"]
    assert "changed" in run["benchmark_context_error"]


def test_metadata_api_failure_does_not_replace_missing_metadata_with_defaults(run, monkeypatch):
    run.pop("benchmark_context")

    async def unavailable(_):
        raise ValueError("Selected cluster is unavailable")

    monkeypatch.setattr(benchmark_context, "_model_pods", unavailable)
    monkeypatch.setattr(benchmark_context, "enrich_model_metadata", AsyncMock())
    asyncio.run(evidence.capture_benchmark_context(run))
    assert "benchmark_context" not in run
    assert run["benchmark_context_error"] == "Selected cluster is unavailable"


def test_missing_deployment_context_never_fails_benchmark_metadata_collection(run):
    run.pop("deployment_configuration")
    asyncio.run(evidence.capture_benchmark_context(run))
    assert run["benchmark_context_error"]
    assert run["benchmark_model_metadata_error"]


def test_probe_is_valid_python():
    compile(benchmark_context._MODEL_PROBE, "<model-probe>", "exec")


def test_probe_never_substitutes_latest_cached_revision_for_unpinned_serving_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=None))

    def unexpected_download(*args, **kwargs):
        pytest.fail("An unpinned model must not select a current cached revision")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=unexpected_download))
    monkeypatch.setattr(sys, "argv", ["probe", "org/model", "org/model", "", ""])
    exec(compile(benchmark_context._MODEL_PROBE, "<model-probe>", "exec"), {})  # noqa: S102


def test_missing_model_source_does_not_inspect_working_directory_or_block_runtime(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unrelated-model"}))
    monkeypatch.chdir(tmp_path)
    device = SimpleNamespace(
        __name__="torch.test_device",
        device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(name="Actual accelerator", total_memory=80 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_device_module=lambda: device, __version__="2.8.0"))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.11.0")
    monkeypatch.setattr(sys, "argv", ["probe", "", "org/model", "", ""])
    exec(compile(benchmark_context._MODEL_PROBE, "<model-probe>", "exec"), {})  # noqa: S102
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == {"model_id": "org/model", "revision": None}
    assert result["hardware"]["gpu_count"] == 1
    assert result["runtime"]["backend"] == "vllm"
    assert result["runtime"]["backend_version"] == "0.11.0"
    assert result["warnings"] == ["model:ValueError"]


def test_batch_failure_rolls_back_every_new_record(run):
    records, _ = evidence.normalize_evaluate_run(run)
    broken = records[0].model_copy(update={"benchmark_id": "invalid-new-record", "timestamp": None})
    dao = BenchmarkRecordDao()
    with pytest.raises(BenchmarkRecordDaoError):
        dao.save_many([records[0], broken])
    assert dao.list() == []


def test_existing_record_is_never_overwritten_by_later_backfill(run):
    records, _ = evidence.normalize_evaluate_run(run)
    dao = BenchmarkRecordDao()
    original = dao.save(records[0])
    sample_metrics(run)["throughput_tps"] = 999.0
    report = evidence.ingest_evaluate_run(run)
    assert report["existing_ids"] == [original.benchmark_id]
    assert not report["inserted_ids"]
    assert dao.get(original.benchmark_id) == original


def test_backfill_does_not_write_to_active_source_records(run, monkeypatch, capsys):
    run["status"] = "running"
    dao = EvaluateRunDao()
    dao.save(EvaluateRunRecord.model_validate(run))
    monkeypatch.setattr(sys, "argv", ["backfill", "--apply", "--run-id", run["id"]])
    evidence.main()
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"
    assert "benchmark_evidence" not in dao.get(run["id"]).model_dump()


@pytest.mark.parametrize("quantization,duplicate", [("none", False), ("awq", False), ("none", True)])
def test_live_probe_reads_actual_metadata_and_refuses_packed_or_duplicate_weights(
    tmp_path, monkeypatch, capsys, quantization, duplicate
):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test",
                "num_experts": 4,
                **({"quantization_config": {"quant_method": quantization}} if quantization != "none" else {}),
            }
        )
    )
    (root / "model.safetensors").touch()
    if duplicate:
        (root / "another.safetensors").touch()
    device = SimpleNamespace(
        __name__="torch.test_device",
        device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(name="Actual accelerator", total_memory=80 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_device_module=lambda: device, __version__="2.8.0"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **kw: root))

    class Header:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return ["tensor"]

        def get_slice(self, key):
            return SimpleNamespace(get_shape=lambda: [2000, 4000])

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **kw: Header()))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.11.0")
    monkeypatch.setattr(sys, "argv", ["probe", str(root), "org/model", "pinned-revision", ""])
    exec(compile(benchmark_context._MODEL_PROBE, "<model-probe>", "exec"), {})  # noqa: S102
    result = json.loads(capsys.readouterr().out)
    if quantization != "none" or duplicate:
        assert "parameter_count_b" not in result["model"]
    else:
        assert result["model"]["parameter_count_b"] == 0.008
    assert result["model"]["architecture"] == "moe"
    assert result["hardware"]["vram_per_gpu_gib"] == 80
    assert result["runtime"]["backend_version"] == "0.11.0"


@pytest.mark.parametrize("missing", [None, "unknown", ""])
def test_missing_auxiliary_metadata_is_archived_as_null_and_roundtrips(run, missing):
    run["benchmark_context"]["model"] = {"model_id": "org/model"}
    run["benchmark_context"]["runtime"] = {"backend": "vllm", "backend_version": missing}
    run["benchmark_context"]["hardware"] = {}
    report = evidence.ingest_evaluate_run(run)
    assert report["status"] == "stored"
    (record,) = BenchmarkRecordDao().list()
    assert record.model.parameter_count_b is None
    assert record.model.architecture is None
    assert record.runtime.backend_version is None
    assert record.hardware.accelerator is None
    assert record.provenance["attribution"] == "serving-snapshot"
    assert "runtime.backend_version" in report["metadata"]["missing_fields"]


def test_saved_configuration_can_archive_without_claiming_historical_hardware(run):
    run.pop("benchmark_context")
    run["benchmark_context_error"] = "Serving-container metadata probe failed"
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    assert records[0].model.model_id == "org/model"
    assert records[0].hardware.gpu_count is None
    assert records[0].hardware.accelerator is None
    assert records[0].runtime.backend_version is None
    assert records[0].provenance["attribution"] == "saved-configuration"


@pytest.mark.parametrize("prefill", [False, True])
def test_operator_hardware_declaration_preserves_source_and_configured_count(run, prefill):
    run.pop("benchmark_context")
    if prefill:
        content = run["deployment_configuration"]["content"]
        content["prefill"] = {"tensorParallelSize": 1, "replicaCount": 1}
        run["deployment_configuration"]["checksum"] = configuration_checksum(content)
    evidence.declare_historical_hardware(run, "Intel Arc Pro B60", 24)
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    record = records[0]
    assert record.hardware.accelerator == "Intel Arc Pro B60"
    assert record.hardware.vram_per_gpu_gib == 24
    assert record.hardware.gpu_count == (2 if prefill else 1)
    assert record.provenance["attribution"] == "saved-configuration"
    declaration = record.provenance["hardware_enrichment"]
    assert declaration["source"] == "operator-declaration"
    assert declaration["gpu_count_source"] == "saved-deployment-configuration"
    run["benchmark_hardware_metadata"]["hardware"]["gpu_count"] = 99
    assert "card count disagrees" in evidence.normalize_evaluate_run(run)[1][0]["reason"]


def test_operator_hardware_cli_previews_then_persists_idempotently(run, monkeypatch, capsys):
    run.pop("benchmark_context")
    dao = EvaluateRunDao()
    dao.save(EvaluateRunRecord.model_validate(run))
    arguments = [
        "backfill",
        "--run-id",
        run["id"],
        "--hardware-accelerator",
        "Intel Arc Pro B60",
        "--hardware-vram-per-gpu-gib",
        "24",
        "--hardware-count-from-configuration",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    evidence.main()
    assert json.loads(capsys.readouterr().out)["status"] == "eligible"
    assert not BenchmarkRecordDao().list()
    assert "benchmark_hardware_metadata" not in dao.get(run["id"]).model_dump()
    monkeypatch.setattr(sys, "argv", arguments + ["--apply"])
    evidence.main()
    assert json.loads(capsys.readouterr().out)["status"] == "stored"
    (record,) = BenchmarkRecordDao().list()
    original = record.model_dump()
    assert record.hardware.accelerator == "Intel Arc Pro B60"
    original_declaration = dao.get(run["id"]).model_dump()["benchmark_hardware_metadata"]
    assert dao.get(run["id"]).model_dump()["benchmark_hardware_metadata"]["source"] == "operator-declaration"
    evidence.main()
    assert json.loads(capsys.readouterr().out)["inserted_ids"] == []
    assert BenchmarkRecordDao().get(record.benchmark_id).model_dump() == original
    assert dao.get(run["id"]).model_dump()["benchmark_hardware_metadata"] == original_declaration


def test_hardware_declarations_refuse_live_snapshots_and_invalid_memory(run):
    with pytest.raises(ValueError, match="without a serving snapshot"):
        evidence.declare_historical_hardware(run, "Intel Arc Pro B60", 24)
    run.pop("benchmark_context")
    with pytest.raises(ValueError, match="finite positive"):
        evidence.declare_historical_hardware(run, "Intel Arc Pro B60", float("nan"))
    evidence.declare_historical_hardware(run, "Intel Arc Pro B60", 24)
    run["benchmark_hardware_metadata"]["configuration_checksum"] = "different"
    assert "does not identify" in evidence.normalize_evaluate_run(run)[1][0]["reason"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--hardware-accelerator", "Intel Arc Pro B60"],
        [
            "--hardware-accelerator",
            "Intel Arc Pro B60",
            "--hardware-vram-per-gpu-gib",
            "24",
            "--hardware-count-from-configuration",
        ],
    ],
)
def test_hardware_cli_requires_explicit_scope_and_complete_declaration(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["backfill", *arguments])
    with pytest.raises(SystemExit) as error:
        evidence.main()
    assert error.value.code == 2
    assert not BenchmarkRecordDao().list()


def test_public_model_api_enriches_descriptors_without_claiming_serving_revision(run, monkeypatch):
    run.pop("benchmark_context")
    sha = "a" * 40
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if "/api/models/" in str(request.url):
            return httpx.Response(200, json={"sha": sha, "safetensors": {"total": 600_000_000}})
        return httpx.Response(200, json={"model_type": "qwen3", "num_experts": 0})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        benchmark_context.httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs)
    )
    asyncio.run(benchmark_context.enrich_model_metadata(run))
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    assert records[0].model.family == "qwen3"
    assert records[0].model.architecture == "dense"
    assert records[0].model.parameter_count_b == 0.6
    assert records[0].model.revision is None
    assert run["benchmark_model_metadata"]["api_revision"] == sha
    assert not run["benchmark_model_metadata"]["serving_revision_pinned"]
    assert f"/resolve/{sha}/config.json" in requests[1]


def test_public_metadata_failure_does_not_block_archival(run, monkeypatch):
    run.pop("benchmark_context")
    client = httpx.AsyncClient
    monkeypatch.setattr(
        benchmark_context.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(lambda request: httpx.Response(503)), **kwargs),
    )
    asyncio.run(benchmark_context.enrich_model_metadata(run))
    assert "HTTPStatusError" in run["benchmark_model_metadata_error"]
    assert evidence.normalize_evaluate_run(run)[0]


def _write_workspace_archive(tmp_path, members):
    """Compress already-built tar members into tmp_path/workspace.tar.zst."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload, kind in members:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "target.json"
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    (tmp_path / "workspace.tar.zst").write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))


def test_native_archive_recovers_measured_tokens_and_stage_concurrency(run, tmp_path):
    run.pop("matrix")
    run.pop("matrix_results")
    run.pop("benchmark_context")
    run["shared_prefix"] = {"system_prompt_len": 12, "question_len": 4}
    metrics = {
        "success_count": 2,
        "failure_count": 0,
        "request_count": 2,
        "token_distributions": {"input": {"mean": 0}, "output": {"mean": 999}},
        "summary_path": str(tmp_path / "benchmark_report,_stage_0_lifecycle_metrics.json.yaml"),
    }
    run["rate_stage_results"] = [{"metrics": metrics}]
    native = {
        "benchmark_time_seconds": 2,
        "successes": {
            "count": 2,
            "prompt_tokens": {"mean": 16, "p95": 16, "total": 32},
            "output_len": {"mean": 8, "p95": 8},
            "output_tokens": {"mean": 10, "total": 20},
            "latency": {"request_latency": {"mean": 0.5, "p95": 0.6}},
            "throughput": {"output_tokens_per_sec": 8, "requests_per_sec": 1},
        },
        "failures": {"count": 0},
    }
    _write_workspace_archive(tmp_path, [("./stage_0_lifecycle_metrics.json", json.dumps(native).encode(), "file")])
    records, issues = evidence.normalize_evaluate_run(run)
    assert not issues
    (record,) = records
    assert record.workload.mean_input_tokens == 16
    assert record.workload.p95_input_tokens == 16
    assert record.workload.mean_output_tokens == 8
    assert record.workload.concurrency == 0.5
    assert record.metrics.e2e_p95_ms == 600
    assert record.metrics.throughput_tokens_per_s == 8
    assert metrics["token_distributions"]["input"]["mean"] == 0  # Original result remains unchanged.
    metrics["request_count"] = 3
    assert "counts disagree" in evidence.normalize_evaluate_run(run)[1][0]["reason"]


def test_read_lifecycle_artifact_reads_archived_member(tmp_path):
    _write_workspace_archive(tmp_path, [("./config.yaml", b"data:\n  type: random\n", "file")])
    assert evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml") == b"data:\n  type: random\n"


def test_read_lifecycle_artifact_rejects_missing_member(tmp_path):
    _write_workspace_archive(tmp_path, [("./other.yaml", b"unrelated", "file")])
    with pytest.raises(ValueError, match="Unable to read config.yaml from its lifecycle archive"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


def test_read_lifecycle_artifact_rejects_duplicate_member(tmp_path):
    _write_workspace_archive(tmp_path, [("./config.yaml", b"first", "file"), ("./config.yaml", b"second", "file")])
    with pytest.raises(ValueError, match="duplicate"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


def test_read_lifecycle_artifact_rejects_non_regular_member(tmp_path):
    _write_workspace_archive(tmp_path, [("./config.yaml", b"", "symlink")])
    with pytest.raises(ValueError, match="not a regular file"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


def test_read_lifecycle_artifact_rejects_member_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "_LIFECYCLE_ARTIFACT_LIMIT", 4)
    _write_workspace_archive(tmp_path, [("./config.yaml", b"12345", "file")])
    with pytest.raises(ValueError, match="exceeds the 64 MiB limit"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


@pytest.mark.parametrize("damage", ["corrupt", "truncated-stream"])
def test_read_lifecycle_artifact_rejects_corrupt_or_truncated_stream(tmp_path, damage):
    _write_workspace_archive(tmp_path, [("./config.yaml", b"data: {}\n", "file")])
    archive = tmp_path / "workspace.tar.zst"
    data = archive.read_bytes()
    archive.write_bytes(b"not a zstd stream" if damage == "corrupt" else data[: len(data) // 2])
    with pytest.raises(ValueError, match="Unable to read config.yaml from its lifecycle archive"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


def test_read_lifecycle_artifact_rejects_archive_truncated_mid_member(tmp_path):
    payload = bytes(range(256)) * 256  # 64 KiB, larger than one tar block
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("./config.yaml")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    raw = buffer.getvalue()
    (tmp_path / "workspace.tar.zst").write_bytes(zstandard.ZstdCompressor().compress(raw[: 512 + len(payload) // 2]))
    with pytest.raises(ValueError, match="Unable to read config.yaml from its lifecycle archive"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


def test_read_lifecycle_artifact_enforces_deadline(tmp_path, monkeypatch):
    _write_workspace_archive(tmp_path, [("./config.yaml", b"data: {}\n", "file")])
    clock = iter([0.0, 16.0])
    monkeypatch.setattr(evidence.time, "monotonic", lambda: next(clock, 16.0))
    with pytest.raises(ValueError, match="Lifecycle archive read timed out"):
        evidence.read_lifecycle_artifact(tmp_path / "summary.json", "config.yaml")


def _artifact_router(monkeypatch, run_record):
    router = importlib.import_module("llm_d_lens.evaluation.router")
    monkeypatch.setattr(router, "_get", lambda kind, run_id: run_record if run_id == run_record["id"] else None)
    return router


def test_run_artifacts_list_and_download_loose_reports(monkeypatch, tmp_path):
    (tmp_path / "per_request_lifecycle_metrics.json").write_text('[{"stage_id": 0}]')
    (tmp_path / "stage_0_lifecycle_metrics.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not a report")
    run_record = {"id": "run-1", "metrics": {"summary_path": str(tmp_path / "summary_lifecycle_metrics.json")}}
    router = _artifact_router(monkeypatch, run_record)

    listing = asyncio.run(router.list_run_artifacts("run-1"))
    assert listing == {
        "items": [
            {"name": "per_request_lifecycle_metrics.json", "size_bytes": 17},
            {"name": "stage_0_lifecycle_metrics.json", "size_bytes": 2},
        ],
        "archived_in": [],
    }
    response = asyncio.run(router.get_run_artifact("run-1", "per_request_lifecycle_metrics.json"))
    assert response.body == b'[{"stage_id": 0}]'
    assert response.media_type == "application/json"


def test_run_artifacts_serve_matrix_points_and_archived_members(monkeypatch, tmp_path):
    point = tmp_path / "matrix-1"
    point.mkdir()
    _write_workspace_archive(point, [("./config.yaml", b"data: {}\n", "file")])
    run_record = {
        "id": "run-1",
        "matrix_results": [{}, {"metrics": {"summary_path": str(point / "summary_lifecycle_metrics.json")}}],
    }
    router = _artifact_router(monkeypatch, run_record)

    assert asyncio.run(router.list_run_artifacts("run-1")) == {"items": [], "archived_in": ["matrix-1"]}
    response = asyncio.run(router.get_run_artifact("run-1", "matrix-1/config.yaml"))
    assert response.body == b"data: {}\n"
    assert response.media_type == "application/yaml"


def test_run_artifacts_reject_unknown_names_runs_and_missing_reports(monkeypatch, tmp_path):
    run_record = {"id": "run-1", "metrics": {"summary_path": str(tmp_path / "summary_lifecycle_metrics.json")}}
    router = _artifact_router(monkeypatch, run_record)

    def status(call):
        with pytest.raises(router.HTTPException) as error:
            asyncio.run(call)
        return error.value.status_code

    assert status(router.get_run_artifact("run-1", "../../etc/passwd")) == 422
    assert status(router.get_run_artifact("run-1", "other.json")) == 422
    assert status(router.get_run_artifact("run-1", "matrix-9/config.yaml")) == 404
    assert status(router.get_run_artifact("run-1", "per_request_lifecycle_metrics.json")) == 404
    assert status(router.list_run_artifacts("missing")) == 404
