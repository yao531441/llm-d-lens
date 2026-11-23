"""Tests for the public deployment query API."""

import asyncio
import importlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from llm_d_lens.deploy.contracts import DeploymentCaseStatus, DeploymentStatus
from llm_d_lens.deploy.providers.hardware_profile import runtime_image
from llm_d_lens.deploy.runtime.composition import build_namespace_factory
from llm_d_lens.deploy.standard_kubernetes_service import (
    StandardKubernetesServiceRequest,
    build_standard_kubernetes_service_configuration,
)
from llm_d_lens.storage.contracts import StorageVolumeKind, StorageVolumePurpose

router = importlib.import_module("llm_d_lens.deploy.router")
executions = importlib.import_module("llm_d_lens.deploy.executions")


@pytest.mark.asyncio
async def test_runtime_classes_api_exposes_capabilities_without_claiming_runtime_readiness(monkeypatch):
    from unittest.mock import AsyncMock

    accesses = []
    monkeypatch.setattr(router, "require_cluster_access", lambda principal, cluster_id: accesses.append(cluster_id))
    monkeypatch.setattr(
        router,
        "discover_runtime_classes",
        AsyncMock(
            return_value={
                "items": [
                    {
                        "metadata": {"name": "nvidia"},
                        "handler": "gpu-handler",
                        "scheduling": {"nodeSelector": {"pool": "gpu"}},
                    }
                ],
                "nodes": [
                    {"metadata": {"name": "gpu-node", "labels": {"pool": "gpu"}}},
                    {"metadata": {"name": "cpu-node"}},
                ],
            }
        ),
    )
    result = await router.list_deployment_runtime_classes("cluster-a", accelerator="nvidia", access_mode="plugin")
    assert accesses == ["cluster-a"]
    assert result["available_suggestion"] == "nvidia"
    assert result["items"][0]["matching_nodes"] == ["gpu-node"]
    assert result["requirement"] == result["node_runtime_readiness"] == "unknown"


@pytest.mark.asyncio
async def test_runtime_classes_api_reports_discovery_failure(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(router, "discover_runtime_classes", AsyncMock(side_effect=ValueError("discovery unavailable")))
    with pytest.raises(HTTPException) as error:
        await router.list_deployment_runtime_classes("cluster-a")
    assert error.value.status_code == 503


def test_deployment_run_route_requires_request_body_not_cluster_query():
    route = next(route for route in router.router.routes if route.path.endswith("/runs") and "POST" in route.methods)
    assert route.endpoint is router.create_run
    assert not [parameter for parameter in route.dependant.query_params if parameter.required]


def test_standard_kubernetes_service_configuration_is_an_immutable_baseline_artifact():
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-8B",
        deployment_name="qwen3-8b-20260904",
        cluster_session_id="a" * 36,
        replicas=2,
        tensor_parallel_size=4,
        storage_type="model-cache",
        storage_volume_id="storage-model-cache",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        use_cluster_proxy=True,
        http_proxy="http://proxy.example:911",
        https_proxy="http://proxy.example:911",
        no_proxy="localhost,.svc",
    )

    configuration = build_standard_kubernetes_service_configuration(request)

    assert configuration.provider_ref == "baseline-vllm"
    assert configuration.type == "baseline"
    assert configuration.content["model"] == {"name": "Qwen/Qwen3-8B"}
    assert configuration.content["decode"] == {
        "replicaCount": 2,
        "tensorParallelSize": 4,
        "maxModelLen": 8192,
    }
    assert configuration.content["runtime"] == {
        "image": runtime_image(),
        "imageMode": "pull-existing",
        "modelSource": "auto-cache",
        "storageVolumeId": "storage-model-cache",
        "cacheOffline": True,
        "environment": {
            "HTTP_PROXY": "http://proxy.example:911",
            "HTTPS_PROXY": "http://proxy.example:911",
            "NO_PROXY": "localhost,.svc,127.0.0.1,.cluster.local",
        },
    }
    assert configuration.provenance["deployment_name"] == "qwen3-8b-20260904"
    assert "namespace_policy" not in configuration.provenance
    deployment_request = SimpleNamespace(
        provenance={"guide_id": configuration.provider_ref},
        deployment_policy=SimpleNamespace(value={"namespace_policy": configuration.provenance.get("namespace_policy")}),
        configuration_artifacts=[SimpleNamespace(content=json.dumps(configuration.content))],
    )
    assert build_namespace_factory("llmd-")(deployment_request).startswith("llmd-baseline-vllm-2-4-")
    assert {item["name"]: item["value"] for item in configuration.content["customParameters"]} == {
        "gpu-memory-utilization": "0.85",
        "enable-prefix-caching": "false",
        "max-num-seqs": "128",
        "max-num-batched-tokens": "4096",
        "enforce-eager": "true",
    }


def test_standard_configuration_uses_selected_accelerator_image(monkeypatch):
    monkeypatch.setenv("PRISM_DEPLOY_ACCELERATOR", "intel-xpu")
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen3-standard",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_volume_id="storage-model-cache",
    )

    configuration = build_standard_kubernetes_service_configuration(request, accelerator="nvidia")

    assert configuration.content["runtime"]["image"] == runtime_image(accelerator="nvidia")
    assert configuration.content["hardware_profile"] == "nvidia"


@pytest.fixture
def standard_cluster_hardware(monkeypatch):
    monkeypatch.setattr(
        "llm_d_lens.cluster.sessions.require_active_session",
        lambda _session_id: SimpleNamespace(server_id="cluster-a"),
    )
    monkeypatch.setattr(
        "llm_d_lens.cluster.registry.require_cluster",
        lambda _cluster_id: SimpleNamespace(id="cluster-a"),
    )

    async def overview(_cluster):
        return {"hardware": {"accelerators": [{"id": "nvidia", "gpuCount": 2}]}}

    monkeypatch.setattr("llm_d_lens.cluster.service.build_overview", overview)

    async def ready_volume(*_args, **_kwargs):
        return SimpleNamespace(kind=StorageVolumeKind.NFS, purposes=[StorageVolumePurpose.MODEL_CACHE])

    async def ready_entries(**_kwargs):
        return [
            SimpleNamespace(
                status="ready",
                source=SimpleNamespace(huggingface=SimpleNamespace(repo_id="Qwen/Qwen3-0.6B", revision="main")),
            )
        ]

    monkeypatch.setattr("llm_d_lens.storage.service.get_ready_volume", ready_volume)
    monkeypatch.setattr("llm_d_lens.model_cache.service.default_service", lambda: SimpleNamespace(list=ready_entries))


def test_standard_kubernetes_service_run_keeps_the_requested_name_in_run_provenance(
    monkeypatch, standard_cluster_hardware
):
    captured = {}

    async def start_run(request, **_kwargs):
        captured["request"] = request
        return request

    monkeypatch.setattr(router.deployment_run_manager, "start_run", start_run)
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen3-standard",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_volume_id="storage-model-cache",
    )

    asyncio.run(router.create_standard_kubernetes_service_run(request))

    assert captured["request"].provenance["deployment_name"] == "qwen3-standard"
    assert captured["request"].configurations[0].content["runtime"]["image"] == runtime_image(accelerator="nvidia")
    assert captured["request"].configurations[0].content["hardware_profile"] == "nvidia"
    assert captured["request"].runtime_binding.placement == {"replicas": "prefer-same-node"}

    request.placement_replicas = "balanced"
    asyncio.run(router.create_standard_kubernetes_service_run(request))
    assert captured["request"].runtime_binding.placement == {"replicas": "balanced"}


def test_standard_run_uses_ready_cache_revision(monkeypatch, standard_cluster_hardware):
    from llm_d_lens.model_cache import service as model_cache_service
    from llm_d_lens.storage import service as storage_service

    async def get_ready_volume(*_args, **_kwargs):
        return SimpleNamespace(kind=StorageVolumeKind.LOCAL_DISK, purposes=[StorageVolumePurpose.MODEL_CACHE])

    async def list_entries(**_kwargs):
        return [
            SimpleNamespace(
                status="ready",
                source=SimpleNamespace(huggingface=SimpleNamespace(repo_id="Qwen/Qwen3-0.6B", revision="v1")),
            )
        ]

    captured = {}

    async def start_run(request, **_kwargs):
        captured["request"] = request
        return request

    monkeypatch.setattr(storage_service, "get_ready_volume", get_ready_volume)
    monkeypatch.setattr(model_cache_service, "default_service", lambda: SimpleNamespace(list=list_entries))
    monkeypatch.setattr(router.deployment_run_manager, "start_run", start_run)
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_type="model-cache",
        storage_volume_id="storage-model-cache",
        use_cluster_proxy=False,
    )

    asyncio.run(router.create_standard_kubernetes_service_run(request))

    assert {"target": "decode", "kind": "argument", "name": "revision", "value": "v1"} in captured[
        "request"
    ].configurations[0].content["customParameters"]


def test_standard_run_restricts_required_colocation_to_selected_node(monkeypatch, standard_cluster_hardware):
    captured = {}

    async def start_run(request, **_kwargs):
        captured["request"] = request
        return request

    async def nodes(*_args, **_kwargs):
        return [{"metadata": {"labels": {"kubernetes.io/hostname": "smc-22"}}}]

    monkeypatch.setattr(router.deployment_run_manager, "start_run", start_run)
    monkeypatch.setattr(
        "llm_d_lens.cluster.sessions.require_active_session",
        lambda _session_id: SimpleNamespace(server_id="cluster-a"),
    )
    monkeypatch.setattr(router, "current_principal", lambda _request: None)
    monkeypatch.setattr(router, "require_cluster_access", lambda *_args: None)
    monkeypatch.setattr(router, "list_resources", nodes)
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen3-standard",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_volume_id="storage-model-cache",
        placement_replicas="same-node",
        placement_node_label={"key": "kubernetes.io/hostname", "value": "smc-22"},
    )

    asyncio.run(router.create_standard_kubernetes_service_run(request))
    assert captured["request"].runtime_binding.placement == {
        "replicas": "same-node",
        "nodeLabel": {"key": "kubernetes.io/hostname", "value": "smc-22"},
    }

    request.placement_node_label = {"key": "kubernetes.io/hostname", "value": "missing"}
    with pytest.raises(HTTPException, match="selected node label is not present"):
        asyncio.run(router.create_standard_kubernetes_service_run(request))


def test_baseline_vllm_renders_boolean_arguments_as_cli_flags():
    from llm_d_lens.deploy.providers.baseline_vllm import BaselineVllmAdapter

    resources = BaselineVllmAdapter._resources(
        {
            "accelerator": "intel-xpu",
            "model": "Qwen/Qwen3-0.6B",
            "image": "ghcr.io/llm-d/llm-d-xpu:v0.9.0",
            "replicas": 1,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "mount_path": "",
            "pvc_name": "",
            "model_source": "huggingface",
            "environment": {},
            "serving_arguments": {},
            "custom_parameters": [
                {"target": "decode", "kind": "argument", "name": "enable-auto-tool-choice", "value": "true"},
                {"target": "decode", "kind": "argument", "name": "enable-prefix-caching", "value": "true"},
                {"target": "decode", "kind": "argument", "name": "enforce-eager", "value": "false"},
                {"target": "decode", "kind": "argument", "name": "max-num-seqs", "value": "256"},
            ],
        }
    )

    args = resources[1]["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--enable-auto-tool-choice" in args
    assert "--enable-prefix-caching" in args
    assert "--no-enforce-eager" in args
    assert "--max-num-seqs=256" in args
    assert not any(argument.startswith("--enable-auto-tool-choice=") for argument in args)
    assert not any(argument.startswith("--enable-prefix-caching=") for argument in args)


def test_baseline_vllm_accepts_the_standard_model_market_namespace():
    from llm_d_lens.deploy.providers.baseline_vllm import BaselineVllmAdapter

    adapter = BaselineVllmAdapter(None, "llmd-", 900, None)

    assert adapter._namespace({"namespace": "standard-4-1-20260904081451502388"}) == "standard-4-1-20260904081451502388"


@pytest.mark.parametrize("storage_type", ["local-cache", "pvc"])
def test_standard_kubernetes_service_rejects_non_model_cache_storage(storage_type):
    with pytest.raises(ValidationError, match="storage_type"):
        StandardKubernetesServiceRequest(
            model="Qwen/Qwen3-0.6B",
            deployment_name="qwen3-pvc",
            cluster_session_id="a" * 36,
            replicas=1,
            tensor_parallel_size=1,
            storage_type=storage_type,
            storage_volume_id="storage-model-cache",
        )


def test_standard_kubernetes_service_configuration_uses_registered_model_cache_storage():
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen3-model-cache",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_type="model-cache",
        storage_volume_id="storage-model-cache",
    )

    configuration = build_standard_kubernetes_service_configuration(request)

    assert configuration.content["runtime"]["storageVolumeId"] == "storage-model-cache"
    assert configuration.content["runtime"]["cacheOffline"] is True
    assert "mountPath" not in configuration.content["runtime"]
    assert "pvcName" not in configuration.content["runtime"]


def test_ready_model_cache_entry_exists_requires_matching_ready_huggingface_model():
    ready_entry = SimpleNamespace(
        status="ready",
        source=SimpleNamespace(huggingface=SimpleNamespace(repo_id="Qwen/Qwen3-0.6B")),
    )
    downloading_entry = SimpleNamespace(
        status="downloading",
        source=SimpleNamespace(huggingface=SimpleNamespace(repo_id="Qwen/Qwen3-0.6B")),
    )

    assert router._ready_model_cache_entry([downloading_entry, ready_entry], "qwen/qwen3-0.6b") is ready_entry
    assert router._ready_model_cache_entry_exists([downloading_entry, ready_entry], "qwen/qwen3-0.6b")


def test_ready_model_cache_entry_exists_rejects_missing_model():
    ready_entry = SimpleNamespace(
        status="ready",
        source=SimpleNamespace(huggingface=SimpleNamespace(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF")),
    )

    assert router._ready_model_cache_entry([ready_entry], "Qwen/Qwen3-0.6B") is None
    assert not router._ready_model_cache_entry_exists([ready_entry], "Qwen/Qwen3-0.6B")


def test_standard_model_cache_uses_downloaded_revision():
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_type="model-cache",
        storage_volume_id="storage-model-cache",
        use_cluster_proxy=False,
    )

    configuration = build_standard_kubernetes_service_configuration(request, model_revision="v1")

    assert {"target": "decode", "kind": "argument", "name": "revision", "value": "v1"} in configuration.content[
        "customParameters"
    ]


def test_standard_model_cache_rejects_conflicting_revision():
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_type="model-cache",
        storage_volume_id="storage-model-cache",
        vllm_arguments=[{"name": "revision", "value": "main"}],
        use_cluster_proxy=False,
    )

    with pytest.raises(ValueError, match="revision does not match"):
        build_standard_kubernetes_service_configuration(request, model_revision="v1")


def test_standard_kubernetes_service_run_rejects_dynamic_pvc_model_cache_storage(monkeypatch):
    import llm_d_lens.cluster.sessions as cluster_sessions
    import llm_d_lens.storage.service as storage_service

    monkeypatch.setattr(
        cluster_sessions,
        "require_active_session",
        lambda _session_id: SimpleNamespace(server_id="cluster-1"),
    )

    async def get_ready_volume(_volume_id, *, cluster_id):
        assert cluster_id == "cluster-1"
        return SimpleNamespace(
            kind=StorageVolumeKind.DYNAMIC_PVC,
            purposes=[StorageVolumePurpose.MODEL_CACHE],
        )

    monkeypatch.setattr(storage_service, "get_ready_volume", get_ready_volume)
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen3-model-cache",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_type="model-cache",
        storage_volume_id="storage-dynamic-pvc",
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(router.create_standard_kubernetes_service_run(request))

    assert captured.value.status_code == 422


def test_get_execution_pods_returns_pods(monkeypatch):
    mock_context = SimpleNamespace(
        execution_id="exec-123",
        namespace="prism-exec-123",
        cluster_id="cluster-1",
        cluster_session_id=None,
    )
    monkeypatch.setattr(router, "find_execution_context", lambda _id: mock_context)

    k8s_pods_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "pod-1",
                        "namespace": "prism-exec-123",
                        "creationTimestamp": "2026-09-06T00:00:00Z",
                    },
                    "status": {
                        "phase": "Running",
                        "podIP": "10.244.0.5",
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "containerStatuses": [
                            {
                                "name": "vllm",
                                "image": "vllm:latest",
                                "ready": True,
                                "restartCount": 0,
                                "state": {"running": {}},
                            }
                        ],
                    },
                    "spec": {"nodeName": "worker-1"},
                }
            ]
        }
    )

    async def mock_run_kubectl(cmd, cluster_id, timeout):
        return SimpleNamespace(returncode=0, stdout=k8s_pods_json, stderr="")

    import llm_d_lens.infrastructure.kubernetes as k8s

    monkeypatch.setattr(k8s, "run_kubectl", mock_run_kubectl)

    res = asyncio.run(router.get_execution_pods("exec-123"))
    assert res["execution_id"] == "exec-123"
    assert res["namespace"] == "prism-exec-123"
    assert len(res["pods"]) == 1
    assert res["pods"][0]["name"] == "pod-1"
    assert res["pods"][0]["phase"] == "Running"
    assert res["pods"][0]["ready"] is True
    assert res["pods"][0]["pod_ip"] == "10.244.0.5"


def test_get_execution_pod_logs_returns_logs(monkeypatch):
    mock_context = SimpleNamespace(
        execution_id="exec-123",
        namespace="prism-exec-123",
        cluster_id="cluster-1",
        cluster_session_id=None,
    )
    monkeypatch.setattr(router, "find_execution_context", lambda _id: mock_context)

    async def mock_run_kubectl(cmd, cluster_id, timeout):
        assert "logs" in cmd
        assert "pod-1" in cmd
        return SimpleNamespace(returncode=0, stdout="[INFO] Server started\n", stderr="")

    import llm_d_lens.infrastructure.kubernetes as k8s

    monkeypatch.setattr(k8s, "run_kubectl", mock_run_kubectl)

    res = asyncio.run(router.get_execution_pod_logs("exec-123", "pod-1", tail=100))
    assert res["execution_id"] == "exec-123"
    assert res["pod_name"] == "pod-1"
    assert res["logs"] == "[INFO] Server started"
    assert res["success"] is True


def test_standard_kubernetes_service_requires_model_cache_volume():
    with pytest.raises(ValidationError, match="storage_volume_id"):
        StandardKubernetesServiceRequest(
            model="Qwen/Qwen3-0.6B",
            deployment_name="qwen3-cache",
            cluster_session_id="a" * 36,
            replicas=1,
            tensor_parallel_size=1,
        )


def test_baseline_vllm_renders_model_cache_pvc():
    from llm_d_lens.deploy.providers.baseline_vllm import BaselineVllmAdapter

    resources = BaselineVllmAdapter._resources(
        {
            "accelerator": "intel-xpu",
            "model": "Qwen/Qwen3-0.6B",
            "image": "ghcr.io/llm-d/llm-d-xpu:v0.9.0",
            "replicas": 1,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "mount_path": "",
            "pvc_name": "shared-model-cache",
            "model_source": "auto-cache",
            "environment": {},
            "serving_arguments": {},
            "custom_parameters": [],
        }
    )

    pod_spec = resources[1]["spec"]["template"]["spec"]
    assert pod_spec["volumes"] == [
        {"name": "model-cache", "persistentVolumeClaim": {"claimName": "shared-model-cache"}},
        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
    ]


def test_standard_kubernetes_service_configuration_inherits_backend_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:911")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:911")
    monkeypatch.setenv("NO_PROXY", "internal.example")
    request = StandardKubernetesServiceRequest(
        model="Qwen/Qwen3-0.6B",
        deployment_name="qwen3-0.6b-20260904",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        storage_volume_id="storage-model-cache",
    )

    configuration = build_standard_kubernetes_service_configuration(request)

    assert configuration.content["runtime"]["environment"] == {
        "HTTP_PROXY": "http://proxy.example:911",
        "HTTPS_PROXY": "http://proxy.example:911",
        "NO_PROXY": "internal.example,localhost,127.0.0.1,.svc,.cluster.local",
    }


@pytest.mark.asyncio
async def test_deployment_run_scheduling_is_idempotent():
    from llm_d_lens.deploy.application import DeploymentRunManager

    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Worker:
        async def execute_run(self, run_id):
            nonlocal calls
            assert run_id == "run-1"
            calls += 1
            started.set()
            await release.wait()

    manager = DeploymentRunManager(store=SimpleNamespace())
    worker = Worker()
    manager._schedule_run("run-1", worker)
    manager._schedule_run("run-1", worker)
    await started.wait()
    assert calls == 1
    release.set()
    await manager._run_tasks["run-1"]


def test_cluster_deployment_source_uses_registered_checkout(monkeypatch, tmp_path):
    from llm_d_lens.deploy import application

    checkout = tmp_path / "llm-d"
    (checkout / "guides").mkdir(parents=True)
    monkeypatch.setattr(
        application.cluster_registry,
        "require_cluster",
        lambda cluster_id: SimpleNamespace(
            id=cluster_id, name="test-cluster", llm_d_repo_path=str(checkout), llm_d_ref="v1.2.3"
        ),
    )

    source = application.cluster_deployment_source(
        SimpleNamespace(server_id="cluster-1"),
        {"kind": "evaluation", "repository": "https://example.com/ignored.git", "branch": "main"},
    )

    assert source == {
        "kind": "evaluation",
        "resolved_repository": str(checkout),
        "ref": "v1.2.3",
        "resolved_from": "cluster-registry",
    }


def test_worker_uses_registered_checkout_for_legacy_cluster_bound_run(monkeypatch):
    from llm_d_lens.deploy import application

    manager = application.DeploymentRunManager(store=SimpleNamespace())
    captured = {}
    monkeypatch.setattr(
        application,
        "cluster_deployment_source",
        lambda session, source: {"resolved_repository": "/checkout/llm-d"},
    )
    monkeypatch.setattr(application, "deployment_runtime_overrides", lambda _session_id: {})
    monkeypatch.setattr(
        manager,
        "_build_worker",
        lambda environment, lifecycle_error=None: captured.update(environment=environment) or SimpleNamespace(),
    )
    # This legacy cluster has no saved proxy config; keep the environment limited
    # to what the run itself supplies.
    monkeypatch.setattr("llm_d_lens.cluster.service.resolve_proxy_env", lambda _cluster_id: {})

    manager._worker(
        SimpleNamespace(
            provenance={
                "cluster_server_id": "cluster-1",
                "cluster_session_id": "session-1",
                "deployment_source": {"kind": "agentic-deployment"},
            }
        )
    )

    assert captured["environment"] == {"LLM_D_ROOT": "/checkout/llm-d"}


def _execution(execution_id: str, status: DeploymentStatus, metadata=None):
    return SimpleNamespace(
        execution_id=execution_id,
        status=status,
        endpoint=(
            SimpleNamespace(url=f"http://{execution_id}.namespace.svc:8000")
            if status == DeploymentStatus.READY
            else None
        ),
        forwarded_endpoint=None,
        namespace="namespace",
        artifact=SimpleNamespace(manifest_ref="vllm/manifest.yaml"),
        provenance={
            "cluster_server_id": "cluster-1",
            "cluster_session_id": "session-1",
            "preserve_deployment": True,
        },
        metadata=metadata,
        created_at=datetime.now(UTC),
        updated_at=None,
    )


def _case(execution_id: str):
    return SimpleNamespace(
        id=f"case-{execution_id}",
        run_id="run-1",
        execution_id=execution_id,
        provider_ref="optimized-baseline",
        status=DeploymentCaseStatus.READY,
        create_request=SimpleNamespace(
            configuration_artifacts=[SimpleNamespace(content='{"model":{"name":"Qwen"}}')],
            deployment_policy=SimpleNamespace(value={"deployment_name": "qwen"}),
        ),
    )


@pytest.mark.asyncio
async def test_list_ready_executions_is_side_effect_free_and_filtered(monkeypatch):
    ready = _execution("execution-ready", DeploymentStatus.READY)
    failed = _execution("execution-failed", DeploymentStatus.FAILED)
    run = SimpleNamespace(
        id="run-1",
        provenance={"cluster_server_id": "cluster-1"},
        cases=[_case(ready.execution_id), _case(failed.execution_id)],
        created_at=datetime.now(UTC),
    )
    store = {ready.execution_id: ready, failed.execution_id: failed}
    monkeypatch.setattr(
        executions,
        "_store",
        SimpleNamespace(list_runs=lambda: [run], get_execution=store.get),
    )

    response = await router.list_executions(
        status=DeploymentStatus.READY,
        cluster_id="cluster-1",
    )

    assert len(response["items"]) == 1
    assert response["items"][0] == {
        "execution_id": "execution-ready",
        "run_id": "run-1",
        "case_id": "case-execution-ready",
        "ownership": "unknown",
        "capabilities": {"connect": False, "update": False, "stop": False, "delete": False},
        "status": "ready",
        "cluster_id": "cluster-1",
        "cluster_session_id": "session-1",
        "namespace": "namespace",
        "name": "qwen",
        "display_name": "",
        "description": "",
        "model": "Qwen",
        "backend": "vLLM",
        "guide": "optimized-baseline",
        "endpoint": "http://execution-ready.namespace.svc:8000",
        "forwarded_endpoint": None,
        "replicas": None,
        "tensor_parallel_size": None,
        "image": None,
        "preserve_deployment": True,
        "created_at": ready.created_at.isoformat(),
        "updated_at": None,
        "max_model_len": None,
        "gpu_memory_utilization": None,
        "enable_prefix_caching": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "storage_type": None,
        "storage_volume_id": None,
        "mount_path": None,
        "pvc_name": None,
        "failure": None,
        "custom_parameters": [],
        "baseline_endpoint": None,
        "service_ref": None,
        "scale": None,
    }


@pytest.mark.asyncio
async def test_execution_payload_exposes_canonical_run_and_case_links(monkeypatch):
    """Execution links let Web and MCP address the same deployment case."""
    ready = _execution("execution-ready", DeploymentStatus.READY)
    run = SimpleNamespace(
        id="run-1",
        provenance={"cluster_server_id": "cluster-1"},
        cases=[_case(ready.execution_id)],
        created_at=datetime.now(UTC),
    )
    monkeypatch.setattr(
        executions,
        "_store",
        SimpleNamespace(list_runs=lambda: [run], get_execution=lambda _id: ready),
    )

    payload = await router.get_execution("execution-ready")

    assert payload["execution_id"] == "execution-ready"
    assert payload["run_id"] == "run-1"
    assert payload["case_id"] == "case-execution-ready"


@pytest.mark.asyncio
async def test_get_execution_reports_missing_execution(monkeypatch):
    monkeypatch.setattr(
        executions,
        "_store",
        SimpleNamespace(list_runs=lambda: [], get_execution=lambda _id: None),
    )

    with pytest.raises(HTTPException) as error:
        await router.get_execution("absent")

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_list_executions_rejects_other_cluster(monkeypatch):
    ready = _execution("execution-ready", DeploymentStatus.READY)
    run = SimpleNamespace(
        id="run-1",
        provenance={"cluster_server_id": "cluster-1"},
        cases=[_case(ready.execution_id)],
        created_at=datetime.now(UTC),
    )
    monkeypatch.setattr(
        executions,
        "_store",
        SimpleNamespace(list_runs=lambda: [run], get_execution=lambda _id: ready),
    )

    response = await router.list_executions(cluster_id="cluster-2")

    assert response == {"items": []}


def test_execution_readable_rejects_a_retired_execution_share():
    from uuid import uuid4

    from llm_d_lens.auth.records import RoleRecord, UserRecord, UserRoleBindingRecord
    from llm_d_lens.auth.service import default_service
    from llm_d_lens.db.dao.cluster import ClusterDao
    from llm_d_lens.db.dao.role import RoleDao, RolePermissionDao, UserRoleBindingDao
    from llm_d_lens.db.dao.user import UserDao

    suffix = uuid4().hex[:6]
    cluster_id = ClusterDao().create(f"dep-{suffix}", "", "apiVersion: v1\nkind: Config").id
    base_role = RoleDao().create(RoleRecord(name=f"base-{suffix}"))
    RolePermissionDao().set_for_role(base_role.id, {"cluster:cluster:read"})
    share_role = RoleDao().create(RoleRecord(name=f"share-{suffix}"))
    RolePermissionDao().set_for_role(share_role.id, {"deployment:run:read"})
    user = UserDao().create(UserRecord(username=f"dep-{suffix}"))
    UserRoleBindingDao().create(
        UserRoleBindingRecord(user_id=user.id, role_id=base_role.id, scope_type="cluster", scope_cluster_id=cluster_id)
    )
    service = default_service()
    UserRoleBindingDao().create(
        UserRoleBindingRecord(
            user_id=user.id,
            role_id=share_role.id,
            scope_type="resource",
            scope_resource_type="deployment_execution",
            scope_resource_id="exec-x",
            scope_cluster_id=cluster_id,
        )
    )
    principal = service.principal_for(user)
    shared = SimpleNamespace(run_id="run-x", execution_id="exec-x", cluster_id=cluster_id)
    assert not router._execution_readable(principal, shared, None)
    unshared = SimpleNamespace(run_id="run-y", execution_id="exec-y", cluster_id=cluster_id)
    assert not router._execution_readable(principal, unshared, None)
