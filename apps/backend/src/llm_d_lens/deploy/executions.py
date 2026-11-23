"""Execution-scoped view of Deploy-owned deployments.

``execution_id`` is the only deployment identifier Deploy exposes to other
modules. Simulation, Monitoring, and the frontend reference a deployment solely
by that id; the ``run_id``/``case_id`` pair stays inside Deploy where it belongs
to run orchestration and audit.

This module owns the reverse lookup from an execution to its owning run/case and
the projection of a deployment into the facts other modules need (namespace,
cluster, endpoints, serving metadata).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_d_lens.configuration.manifest_facts import vllm_argument_value, workload_roles
from llm_d_lens.deploy.application import deployment_run_manager
from llm_d_lens.deploy.contracts import (
    DeploymentMetadata,
    DeploymentMetadataUpdateRequest,
    DeploymentScaleRequest,
    DeploymentStatus,
    utcnow,
)
from llm_d_lens.deploy.data_plane import deployment_uses_shared_gateway
from llm_d_lens.deploy.runtime.composition import RuntimeConfigurationError
from llm_d_lens.deploy.usage import DeploymentInUseError, active_usage_reason
from llm_d_lens.infrastructure.kubernetes import run_kubectl

_store = deployment_run_manager.store
# Serializes user-driven mutations of one deployment so a metadata edit and a
# delete never interleave their read-modify-write cycles.
_execution_locks: dict[str, asyncio.Lock] = {}


class DeploymentExecutionNotFoundError(Exception):
    """Raised when no deployment execution matches the requested id."""


class DeploymentExecutionConflictError(Exception):
    """Raised when a deployment cannot be changed in its current state."""

    def __init__(self, message: str, code: str = "deployment_conflict") -> None:
        super().__init__(message)
        self.code = code


class DeploymentScaleError(Exception):
    """Raised when model-server replicas cannot be read or changed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "deployment_scale_failed",
        status_code: int = 422,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _execution_lock(execution_id: str) -> asyncio.Lock:
    lock = _execution_locks.get(execution_id)
    if lock is None:
        lock = asyncio.Lock()
        _execution_locks[execution_id] = lock
    return lock


@dataclass(frozen=True)
class DeploymentExecutionContext:
    """Deployment facts other modules may consume, keyed by ``execution_id``.

    ``run_id`` and ``case_id`` are carried for Deploy-internal callers only and
    are deliberately excluded from the HTTP projection.
    """

    execution_id: str
    run_id: str
    case_id: str
    status: str
    cluster_id: str | None
    cluster_session_id: str | None
    namespace: str | None
    name: str | None
    display_name: str
    description: str
    model: str | None
    backend: str | None
    guide: str | None
    endpoint: str | None
    forwarded_endpoint: str | None
    replicas: int | None
    tensor_parallel_size: int | None
    image: str | None
    preserve_deployment: bool
    created_at: Any
    updated_at: Any = None
    max_model_len: int | None = None
    gpu_memory_utilization: Any = None
    enable_prefix_caching: Any = None
    max_num_seqs: Any = None
    max_num_batched_tokens: Any = None
    storage_type: str | None = None
    storage_volume_id: str | None = None
    mount_path: str | None = None
    pvc_name: str | None = None
    failure: dict[str, Any] | None = None
    custom_parameters: list[dict[str, Any]] | None = None
    baseline_endpoint: str | None = None
    service_ref: str | None = None
    #: True when an evaluation created this ephemeral deployment for benchmarking.
    evaluate_owned: bool = False
    #: True when a benchmark must reach this deployment through the shared Gateway.
    uses_shared_gateway: bool = False
    configuration_artifact_ids: tuple[str, ...] = ()
    scale: dict[str, Any] | None = None

    @property
    def expected_model_pods(self) -> int:
        if self.scale:
            slices = int(self.scale.get("slices") or 1)
            total = slices * (int(self.scale.get("prefillReplicas") or 0) + int(self.scale.get("decodeReplicas") or 0))
            if total > 0:
                return total
        return max(int(self.replicas or 1), 1)

    def api_payload(self) -> dict[str, Any]:
        """Project the execution for HTTP clients without leaking run/case ids."""
        created_at = self.created_at
        return {
            "execution_id": self.execution_id,
            "status": self.status,
            "cluster_id": self.cluster_id,
            "cluster_session_id": self.cluster_session_id,
            "namespace": self.namespace,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "model": self.model,
            "backend": self.backend,
            "guide": self.guide,
            "endpoint": self.endpoint,
            "forwarded_endpoint": self.forwarded_endpoint,
            "replicas": self.replicas,
            "tensor_parallel_size": self.tensor_parallel_size,
            "image": self.image,
            "preserve_deployment": self.preserve_deployment,
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
            "updated_at": _isoformat(self.updated_at),
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "storage_type": self.storage_type,
            "storage_volume_id": self.storage_volume_id,
            "mount_path": self.mount_path,
            "pvc_name": self.pvc_name,
            "failure": self.failure,
            "custom_parameters": self.custom_parameters or [],
            "baseline_endpoint": self.baseline_endpoint,
            "service_ref": self.service_ref,
            "scale": self.scale,
        }

    def search_haystack(self) -> str:
        """Lower-cased text a deployment can be searched by."""
        return " ".join(
            value.lower()
            for value in (
                self.execution_id,
                self.display_name,
                self.description,
                self.name,
                self.model,
                self.namespace,
                self.guide,
                self.backend,
                self.status,
            )
            if value
        )


def _isoformat(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _rendered_serving_metadata(execution) -> dict[str, Any]:
    if execution is None:
        return {}
    try:
        workloads = workload_roles(_execution_manifest_documents(execution))
    except (OSError, DeploymentScaleError):
        return {}
    metadata: dict[str, Any] = {}
    for workload in workloads:
        role = workload["role"]
        if role not in {"serving", "decode", "prefill"}:
            continue
        for container in workload["pod"].get("containers") or []:
            if container.get("name") != "modelserver":
                continue
            try:
                values = {
                    field: vllm_argument_value(container, {"--" + field.replace("_", "-")}, workload["name"])
                    for field in (
                        "tensor_parallel_size",
                        "max_model_len",
                        "gpu_memory_utilization",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "enable_prefix_caching",
                    )
                }
                for field in ("tensor_parallel_size", "max_model_len"):
                    if values[field] is not None:
                        values[field] = int(values[field])
                prefix_caching = values["enable_prefix_caching"]
                values["enable_prefix_caching"] = (
                    {"": True, "true": True, "false": False}.get(prefix_caching.lower())
                    if prefix_caching is not None
                    else None
                )
                if vllm_argument_value(container, {"--no-enable-prefix-caching"}, workload["name"]) is not None:
                    values["enable_prefix_caching"] = False
            except ValueError:
                continue
            if role == "prefill":
                metadata["prefill_replicas"] = workload["replicas"]
            elif "replicas" not in metadata:
                metadata.update(values)
                metadata.update(backend="vLLM", image=container.get("image"), replicas=workload["replicas"])
    return metadata


def serving_metadata(case, execution) -> dict[str, Any]:
    """Read serving details out of a case's rendered configuration artifact.

    Shared by the execution projection and the deployment payloads returned by
    the run/case routes so a deployment reports the same serving facts wherever
    it is observed.
    """
    content: Any = {}
    if case and case.create_request.configuration_artifacts:
        try:
            content = json.loads(case.create_request.configuration_artifacts[0].content or "{}")
        except json.JSONDecodeError:
            content = {}
    model = content.get("model") if isinstance(content, dict) else {}
    decode = content.get("decode") if isinstance(content, dict) else {}
    prefill = content.get("prefill") if isinstance(content, dict) else {}
    runtime = content.get("runtime") if isinstance(content, dict) else {}
    custom_params = (
        content.get("customParameters")
        if isinstance(content, dict) and isinstance(content.get("customParameters"), list)
        else []
    )

    params_map = {}
    for p in custom_params:
        if isinstance(p, dict) and p.get("name"):
            params_map[p["name"]] = p.get("value")

    manifest_ref = str(execution.artifact.manifest_ref or "") if execution else ""
    provider_ref = case.provider_ref if case else ""
    policy = case.create_request.deployment_policy.value if case else {}
    deployment_name = str((policy or {}).get("deployment_name") or "") or provider_ref
    backend = (
        "sglang"
        if "sglang" in manifest_ref or "sglang" in provider_ref
        else "vLLM"
        if "vllm" in manifest_ref or provider_ref == "optimized-baseline"
        else None
    )
    storage_type = (
        runtime.get("storageType")
        or (
            "model-cache"
            if "storageVolumeId" in runtime
            else "pvc"
            if "pvcName" in runtime
            else "local-cache"
            if "mountPath" in runtime
            else None
        )
        if isinstance(runtime, dict)
        else None
    )
    metadata = {
        "name": deployment_name or None,
        "guide": provider_ref or None,
        "backend": backend,
        "model_name": model.get("name") if isinstance(model, dict) else None,
        "replicas": decode.get("replicaCount") if isinstance(decode, dict) else None,
        "prefill_replicas": prefill.get("replicaCount") if isinstance(prefill, dict) else None,
        "tensor_parallel_size": decode.get("tensorParallelSize") if isinstance(decode, dict) else None,
        "max_model_len": decode.get("maxModelLen") if isinstance(decode, dict) else None,
        "image": runtime.get("image") if isinstance(runtime, dict) else None,
        "storage_type": storage_type,
        "storage_volume_id": runtime.get("storageVolumeId") if isinstance(runtime, dict) else None,
        "mount_path": runtime.get("mountPath") if isinstance(runtime, dict) else None,
        "pvc_name": runtime.get("pvcName") if isinstance(runtime, dict) else None,
        "gpu_memory_utilization": params_map.get("gpu-memory-utilization"),
        "enable_prefix_caching": params_map.get("enable-prefix-caching"),
        "max_num_seqs": params_map.get("max-num-seqs"),
        "max_num_batched_tokens": params_map.get("max-num-batched-tokens"),
        "custom_parameters": custom_params,
    }
    for field, value in _rendered_serving_metadata(execution).items():
        if metadata.get(field) is None:
            metadata[field] = value
    return metadata


def _manifest_disaggregated_set_slices(execution: Any | None) -> int | None:
    if execution is None:
        return None
    artifact = getattr(execution, "artifact", None)
    reference = Path(getattr(artifact, "manifest_ref", None) or "")
    if reference.is_dir():
        reference = reference / "manifest.yaml"
    if not reference.is_file():
        return None
    try:
        manifest = reference.read_text(encoding="utf-8")
    except OSError:
        return None
    if "kind: DisaggregatedSet" not in manifest:
        return None
    try:
        for document in yaml.safe_load_all(manifest):
            if not isinstance(document, dict) or document.get("kind") != "DisaggregatedSet":
                continue
            spec = document.get("spec")
            if isinstance(spec, dict):
                slices = spec.get("slices")
                if isinstance(slices, int) and slices >= 1:
                    return slices
    except yaml.YAMLError:
        return None
    return None


def _scale_projection(serving: dict[str, Any], execution: Any | None) -> dict[str, Any] | None:
    provenance = getattr(execution, "provenance", None) or {}
    manual = provenance.get("manual_scale") if isinstance(provenance, dict) else None
    manual = manual if isinstance(manual, dict) else {}
    decode = manual.get("decodeReplicas")
    if decode is None:
        decode = serving.get("replicas")
    prefill = manual.get("prefillReplicas")
    if prefill is None:
        prefill = serving.get("prefill_replicas")
    if decode is None and prefill is None:
        return None
    slices = manual.get("slices")
    if slices is None:
        contract = {}
        artifact = getattr(execution, "artifact", None)
        payload = getattr(artifact, "rendered_payload", None)
        value = getattr(payload, "value", None)
        if isinstance(value, dict) and isinstance(value.get("deployment_contract"), dict):
            contract = value["deployment_contract"]
        disaggregated_set = contract.get("disaggregatedSet")
        if isinstance(disaggregated_set, dict):
            slices = disaggregated_set.get("slices")
    if slices is None:
        slices = _manifest_disaggregated_set_slices(execution)
    try:
        slices = max(int(slices or 1), 1)
    except (TypeError, ValueError):
        slices = 1
    return {
        "prefillReplicas": prefill,
        "decodeReplicas": decode,
        "slices": slices,
        "manual": bool(manual),
    }


def _context(run, case, execution) -> DeploymentExecutionContext:
    serving = serving_metadata(case, execution)
    scale = _scale_projection(serving, execution)
    metadata = getattr(execution, "metadata", None) or DeploymentMetadata()
    exec_prov = getattr(execution, "provenance", {}) or {}
    run_prov = getattr(run, "provenance", {}) or {}
    cluster_id = _optional_str(exec_prov.get("cluster_server_id")) or _optional_str(run_prov.get("cluster_server_id"))
    cluster_session_id = _optional_str(exec_prov.get("cluster_session_id")) or _optional_str(
        run_prov.get("cluster_session_id")
    )

    failure_obj = getattr(case, "failure", None)
    failure_dict = None
    if failure_obj:
        failure_dict = {
            "title": getattr(failure_obj, "title", None),
            "detail": getattr(failure_obj, "detail", None),
            "code": getattr(failure_obj, "code", None),
            "status": getattr(failure_obj, "status", None),
        }

    endpoint_obj = getattr(execution, "endpoint", None)
    baseline_endpoint = getattr(endpoint_obj, "baseline_url", None) if endpoint_obj else None
    service_ref = getattr(endpoint_obj, "service_ref", None) if endpoint_obj else None

    return DeploymentExecutionContext(
        configuration_artifact_ids=tuple(
            getattr(getattr(execution, "artifact", None), "configuration_artifact_ids", ())
        ),
        execution_id=execution.execution_id,
        run_id=run.id if run else "",
        case_id=case.id if case else "",
        status=execution.status.value if hasattr(execution.status, "value") else str(execution.status),
        cluster_id=cluster_id,
        cluster_session_id=cluster_session_id,
        namespace=execution.namespace,
        name=serving["name"],
        display_name=metadata.display_name,
        description=metadata.description,
        model=serving["model_name"],
        backend=serving["backend"],
        guide=serving["guide"],
        endpoint=execution.endpoint.url if execution.endpoint else None,
        forwarded_endpoint=execution.forwarded_endpoint,
        replicas=(scale or {}).get("decodeReplicas") or serving["replicas"],
        tensor_parallel_size=serving["tensor_parallel_size"],
        image=serving["image"],
        preserve_deployment=bool(execution.provenance.get("preserve_deployment")),
        created_at=getattr(execution, "created_at", None) or getattr(run, "created_at", None),
        updated_at=getattr(execution, "updated_at", None),
        max_model_len=serving["max_model_len"],
        gpu_memory_utilization=serving["gpu_memory_utilization"],
        enable_prefix_caching=serving["enable_prefix_caching"],
        max_num_seqs=serving["max_num_seqs"],
        max_num_batched_tokens=serving["max_num_batched_tokens"],
        storage_type=serving["storage_type"],
        storage_volume_id=serving["storage_volume_id"],
        mount_path=serving["mount_path"],
        pvc_name=serving["pvc_name"],
        failure=failure_dict,
        custom_parameters=serving["custom_parameters"],
        baseline_endpoint=baseline_endpoint,
        service_ref=service_ref,
        evaluate_owned=bool(
            exec_prov.get("evaluate_workflow")
            or exec_prov.get("evaluation_id")
            or exec_prov.get("evaluation_case_id")
            or run_prov.get("evaluate_workflow")
            or run_prov.get("evaluation_id")
        ),
        uses_shared_gateway=deployment_uses_shared_gateway(execution),
        scale=scale,
    )


def _optional_str(value: object | None) -> str | None:
    return str(value) if value else None


def list_execution_contexts(
    *,
    status: DeploymentStatus | None = None,
    statuses: list[DeploymentStatus] | None = None,
    cluster_id: str = "",
    query: str = "",
    limit: int | None = None,
    allowed_cluster_ids: set[str] | None = None,
) -> list[DeploymentExecutionContext]:
    """List deployment executions without creating endpoint port-forwards."""
    wanted = {item for item in (statuses or []) if item is not None}
    if status is not None:
        wanted.add(status)
    needle = query.strip().lower()
    contexts: list[DeploymentExecutionContext] = []
    executions_by_request_id = {}
    if hasattr(_store, "list_executions"):
        try:
            executions_by_request_id = {
                e.request_id: e for e in _store.list_executions() if getattr(e, "request_id", None)
            }
        except Exception:
            executions_by_request_id = {}

    for run in reversed(_store.list_runs()):
        if cluster_id and str(run.provenance.get("cluster_server_id") or "") != cluster_id:
            continue
        if (
            allowed_cluster_ids is not None
            and str(run.provenance.get("cluster_server_id") or "") not in allowed_cluster_ids
        ):
            continue
        for case in run.cases:
            if (
                not case.execution_id
                and case.create_request
                and case.create_request.request_id in executions_by_request_id
            ):
                case.execution_id = executions_by_request_id[case.create_request.request_id].execution_id
            if not case.execution_id:
                continue
            execution = _store.get_execution(case.execution_id)
            if execution is None or (wanted and execution.status not in wanted):
                continue
            context = _context(run, case, execution)
            if needle and needle not in context.search_haystack():
                continue
            contexts.append(context)
    contexts.sort(key=lambda context: context.execution_id)
    contexts.sort(key=lambda context: str(context.created_at or ""), reverse=True)
    return contexts[:limit] if limit is not None else contexts


def find_execution_context(execution_id: str) -> DeploymentExecutionContext | None:
    """Resolve one deployment execution and its owning run/case."""
    execution = _store.get_execution(execution_id)
    if execution is None and execution_id.startswith("pending:"):
        parts = execution_id.split(":", 2)
        if len(parts) == 3:
            run_id, case_id = parts[1], parts[2]
            case = _store.get_case(run_id, case_id)
            if case:
                if case.execution_id:
                    execution = _store.get_execution(case.execution_id)
                    if execution:
                        execution_id = case.execution_id
                if execution is None and case.create_request and hasattr(_store, "list_executions"):
                    req_id = case.create_request.request_id
                    try:
                        execution = next(
                            (e for e in _store.list_executions() if getattr(e, "request_id", None) == req_id), None
                        )
                    except Exception:
                        execution = None
                    if execution:
                        case.execution_id = execution.execution_id
                        execution_id = execution.execution_id

    if execution is not None:
        for run in reversed(_store.list_runs()):
            for case in run.cases:
                if case.execution_id == execution_id or (
                    case.create_request and case.create_request.request_id == execution.request_id
                ):
                    if not case.execution_id:
                        case.execution_id = execution.execution_id
                    return _context(run, case, execution)
        serving = serving_metadata(None, execution)
        scale = _scale_projection(serving, execution)
        metadata = getattr(execution, "metadata", None) or DeploymentMetadata()
        return DeploymentExecutionContext(
            configuration_artifact_ids=tuple(
                getattr(getattr(execution, "artifact", None), "configuration_artifact_ids", ())
            ),
            execution_id=execution.execution_id,
            run_id="",
            case_id="",
            status=execution.status.value if hasattr(execution.status, "value") else str(execution.status),
            cluster_id=_optional_str(execution.provenance.get("cluster_server_id")),
            cluster_session_id=_optional_str(execution.provenance.get("cluster_session_id")),
            namespace=execution.namespace,
            name=serving["name"],
            display_name=metadata.display_name,
            description=metadata.description,
            model=serving["model_name"],
            backend=serving["backend"],
            guide=serving["guide"],
            endpoint=execution.endpoint.url if execution.endpoint else None,
            forwarded_endpoint=execution.forwarded_endpoint,
            replicas=(scale or {}).get("decodeReplicas") or serving["replicas"],
            tensor_parallel_size=serving["tensor_parallel_size"],
            image=serving["image"],
            preserve_deployment=bool(execution.provenance.get("preserve_deployment")),
            created_at=getattr(execution, "created_at", None),
            updated_at=getattr(execution, "updated_at", None),
            scale=scale,
        )

    if execution_id.startswith("pending:"):
        parts = execution_id.split(":", 2)
        if len(parts) == 3:
            run_id, case_id = parts[1], parts[2]
            run = _store.get_run(run_id)
            case = _store.get_case(run_id, case_id)
            if run and case:
                provenance = case.create_request.provenance or run.provenance or {}
                serving = serving_metadata(case, None)
                scale = _scale_projection(serving, None)
                metadata = DeploymentMetadata(description=str(provenance.get("description") or "").strip())
                return DeploymentExecutionContext(
                    execution_id=execution_id,
                    run_id=run.id,
                    case_id=case.id,
                    status=case.status.value if hasattr(case.status, "value") else str(case.status),
                    cluster_id=_optional_str(provenance.get("cluster_server_id")),
                    cluster_session_id=_optional_str(provenance.get("cluster_session_id")),
                    namespace=_optional_str(provenance.get("deployment_namespace")),
                    name=serving["name"] or provenance.get("deployment_name"),
                    display_name=metadata.display_name or str(provenance.get("deployment_name") or ""),
                    description=metadata.description,
                    model=serving["model_name"],
                    backend=serving["backend"],
                    guide=serving["guide"],
                    endpoint=None,
                    forwarded_endpoint=None,
                    replicas=(scale or {}).get("decodeReplicas") or serving["replicas"],
                    tensor_parallel_size=serving["tensor_parallel_size"],
                    image=serving["image"],
                    preserve_deployment=bool(provenance.get("preserve_deployment")),
                    created_at=getattr(run, "created_at", None),
                    updated_at=getattr(run, "finished_at", None),
                    scale=scale,
                )

    return None


def resolve_legacy_execution_id(run_id: str | None, case_id: str | None) -> str | None:
    """Translate a historical ``run_id``/``case_id`` pair into an execution id.

    Only for reading records persisted before ``execution_id`` became the sole
    cross-module deployment reference.
    """
    if not run_id or not case_id:
        return None
    case = _store.get_case(run_id, case_id)
    return case.execution_id if case else None


_TERMINAL_STATUSES = {DeploymentStatus.CLEANED, DeploymentStatus.CLEANED_UP}


def list_storage_volume_usage(volume_id: str) -> list[str]:
    """Return execution ids of non-terminal executions referencing a storage volume.

    Used by the Storage module to reject deleting a volume that is still
    mounted by a deployment. This is a read-only cross-module query (the
    inverse of ``deploy.usage``'s probe registry: here Storage queries Deploy
    directly since Deploy already depends on Storage for mount resolution).
    """
    matches: list[str] = []
    for run in _store.list_runs():
        for case in run.cases:
            if not case.execution_id:
                continue
            execution = _store.get_execution(case.execution_id)
            if execution is None or execution.status in _TERMINAL_STATUSES:
                continue
            runtime = _case_runtime(case)
            if str(runtime.get("storageVolumeId") or "") == volume_id:
                matches.append(case.execution_id)
    return matches


def _case_runtime(case) -> dict[str, Any]:
    if not case.create_request.configuration_artifacts:
        return {}
    try:
        content = json.loads(case.create_request.configuration_artifacts[0].content or "{}")
    except json.JSONDecodeError:
        return {}
    runtime = content.get("runtime") if isinstance(content, dict) else None
    return runtime if isinstance(runtime, dict) else {}


def get_execution_context(execution_id: str) -> DeploymentExecutionContext:
    """Resolve a deployment execution or raise :class:`DeploymentExecutionNotFoundError`.

    This is the supported entry point for other backend modules (Monitoring,
    Simulation) that need deployment facts. Callers must not read the Deploy run
    store directly.
    """
    context = find_execution_context(execution_id)
    if context is None:
        raise DeploymentExecutionNotFoundError(f"deployment execution not found: {execution_id}")
    return context


def execution_deployment_plan(context: DeploymentExecutionContext) -> dict[str, Any]:
    """Return the stored configuration plan and rendered manifest of one execution."""
    case = _store.get_case(context.run_id, context.case_id) if context.run_id and context.case_id else None
    artifacts = case.create_request.configuration_artifacts if case and case.create_request else []
    execution = None if context.execution_id.startswith("pending:") else _store.get_execution(context.execution_id)
    manifest = None
    if execution is not None:
        reference = Path(getattr(execution.artifact, "manifest_ref", None) or "")
        if reference.is_dir():
            reference = reference / "manifest.yaml"
        if reference.is_file():
            manifest = _redact_manifest_secrets(reference.read_text(encoding="utf-8"))
    return {
        "execution_id": context.execution_id,
        "namespace": context.namespace,
        "configuration_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "provider_ref": artifact.provider_ref,
                "schema_version": artifact.schema_version,
                "media_type": artifact.media_type,
                "content": _redact_artifact_content(artifact.content),
            }
            for artifact in artifacts
        ],
        "manifest": manifest,
    }


def _redact_manifest_secrets(manifest: str) -> str | None:
    """Redact Secret values, including Secrets nested in ``List.items``; None when unparsable."""
    try:
        documents = list(yaml.safe_load_all(manifest))
    except yaml.YAMLError:
        return None
    redacted = False
    for document in documents:
        redacted = _redact_secret_values(document) or redacted
    if not redacted:
        return manifest
    return yaml.safe_dump_all([document for document in documents if document is not None], sort_keys=False)


def _redact_secret_values(document: Any) -> bool:
    if not isinstance(document, dict):
        return False
    redacted = False
    if document.get("kind") == "Secret":
        for field in ("data", "stringData"):
            if isinstance(document.get(field), dict) and document[field]:
                document[field] = dict.fromkeys(document[field], "<redacted>")
                redacted = True
    items = document.get("items")
    if isinstance(items, list):
        for item in items:
            redacted = _redact_secret_values(item) or redacted
    return redacted


def _redact_artifact_content(content: str | None) -> str | None:
    if not content or "Secret" not in content:
        return content
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return _redact_manifest_secrets(content) if _is_secret_text(content) else content
    return json.dumps(_redact_embedded_manifests(value))


def _redact_embedded_manifests(value: Any) -> Any:
    if isinstance(value, dict):
        _redact_secret_values(value)
        return {key: _redact_embedded_manifests(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_embedded_manifests(item) for item in value]
    if isinstance(value, str) and _is_secret_text(value):
        return _redact_manifest_secrets(value) or "<redacted>"
    return value


def _is_secret_text(value: str) -> bool:
    return re.search(r"\bkind['\"]?\s*:\s*['\"]?Secret\b", value) is not None


def _execution_cluster_id(execution) -> str | None:
    provenance = execution.provenance or {}
    cluster_ref = provenance.get("cluster_ref")
    if isinstance(cluster_ref, dict) and cluster_ref.get("id"):
        return str(cluster_ref["id"])
    return _optional_str(provenance.get("cluster_server_id"))


def _scale_role(role: str) -> str | None:
    if role == "serving":
        return "decode"
    return role if role in {"prefill", "decode"} else None


def _has_model_server(pod: dict[str, Any]) -> bool:
    for container in pod.get("containers") or []:
        name = str(container.get("name") or "").lower()
        if name in {"modelserver", "model-server", "vllm", "sglang"} or name.startswith("modelserver"):
            return True
        command = container.get("command") or []
        args = container.get("args") or []
        if isinstance(command, str):
            command = [command]
        if isinstance(args, str):
            args = [args]
        invocation = " ".join(str(item) for item in [*command, *args]).lower()
        if "vllm serve" in invocation or "sglang" in invocation:
            return True
    return False


def _execution_manifest_documents(execution) -> list[dict[str, Any]]:
    reference = Path(execution.artifact.manifest_ref or "")
    if reference.is_dir():
        reference = reference / "manifest.yaml"
    if not reference.is_file():
        return []
    try:
        return [
            document
            for document in yaml.safe_load_all(reference.read_text(encoding="utf-8"))
            if isinstance(document, dict)
        ]
    except yaml.YAMLError as error:
        raise DeploymentScaleError(f"rendered deployment manifest is invalid YAML: {error}") from error


async def _live_deployments(execution) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cluster_id = _execution_cluster_id(execution)
    if not execution.namespace or not cluster_id:
        return [], {}
    try:
        result = await run_kubectl(
            ["get", "deployments", "--namespace", execution.namespace, "--output", "json"],
            timeout=10,
            cluster_id=cluster_id,
        )
    except (FileNotFoundError, TimeoutError, OSError):
        return [], {}
    if result.returncode != 0:
        return [], {}
    try:
        items = json.loads(result.stdout or "{}").get("items") or []
    except (TypeError, ValueError):
        return [], {}
    replicas: dict[str, int] = {}
    for item in items:
        name = str((item.get("metadata") or {}).get("name") or "")
        value = (item.get("spec") or {}).get("replicas")
        if name and isinstance(value, int):
            replicas[name] = value
    return items, replicas


async def _live_disaggregated_sets(execution, names: set[str]) -> dict[str, dict[str, Any]]:
    cluster_id = _execution_cluster_id(execution)
    if not execution.namespace or not cluster_id:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        try:
            response = await run_kubectl(
                ["get", "disaggregatedset", name, "--namespace", execution.namespace, "--output", "json"],
                timeout=10,
                cluster_id=cluster_id,
            )
        except (FileNotFoundError, TimeoutError, OSError):
            continue
        if response.returncode != 0:
            continue
        try:
            document = json.loads(response.stdout or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(document, dict) and document.get("kind") == "DisaggregatedSet":
            result[name] = document
    return result


def _scale_workloads(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for item in workload_roles(documents):
        if not _has_model_server(item.get("pod") or {}):
            continue
        role = _scale_role(str(item.get("role") or ""))
        if role is None:
            continue
        name = str(item.get("name") or "")
        kind = str(item.get("kind") or "")
        if not name:
            continue
        if kind not in {"Deployment", "DisaggregatedSet"}:
            unsupported.append(f"{kind}/{name}")
            continue
        role_document = item.get("role_document") or {}
        scaling = role_document.get("scaling") if isinstance(role_document, dict) else None
        workloads.append(
            {
                "role": role,
                "name": name,
                "kind": kind,
                "manifestReplicas": item.get("replicas"),
                "roleIndex": item.get("role_index"),
                "slices": item.get("slices", 1),
                "scalingMode": (scaling or {}).get("mode", "Static") if isinstance(scaling, dict) else "Static",
            }
        )
    if unsupported:
        raise DeploymentScaleError(
            "Manual scaling supports Deployment and DisaggregatedSet model servers; "
            f"unsupported workloads: {', '.join(unsupported)}.",
            code="deployment_scale_unsupported",
        )
    return workloads


def _role_replicas(workloads: list[dict[str, Any]]) -> int | None:
    values = [
        workload["currentReplicas"]
        for workload in workloads
        if isinstance(workload.get("currentReplicas"), int) and workload["currentReplicas"] >= 0
    ]
    return max(values) if values else None


async def _build_scale_plan(execution) -> dict[str, Any]:
    if execution.status not in {DeploymentStatus.READY, DeploymentStatus.FAILED}:
        raise DeploymentScaleError(
            "Deployment must be ready or failed before replicas can be scaled.",
            code="deployment_scale_conflict",
            status_code=409,
        )
    if not execution.namespace or not _execution_cluster_id(execution):
        raise DeploymentScaleError(
            "Deployment namespace or cluster is unavailable.",
            code="deployment_scale_unsupported",
        )
    documents = _execution_manifest_documents(execution)
    live_items, live_replicas = await _live_deployments(execution)
    if not documents:
        documents = live_items
    workloads = _scale_workloads(documents)
    disaggregated_sets = sorted({workload["name"] for workload in workloads if workload["kind"] == "DisaggregatedSet"})
    if len(disaggregated_sets) > 1:
        raise DeploymentScaleError(
            "Manual scaling supports one DisaggregatedSet per execution.",
            code="deployment_scale_unsupported",
        )
    live_sets = await _live_disaggregated_sets(execution, set(disaggregated_sets))
    resolved: list[dict[str, Any]] = []
    for workload in workloads:
        if workload["kind"] != "DisaggregatedSet":
            workload["currentReplicas"] = live_replicas.get(str(workload["name"]), workload.get("manifestReplicas"))
            resolved.append(workload)
    for name in disaggregated_sets:
        live_document = live_sets.get(name)
        if live_document is not None:
            resolved.extend(
                workload for workload in _scale_workloads([live_document]) if workload["kind"] == "DisaggregatedSet"
            )
        else:
            resolved.extend(
                workload
                for workload in workloads
                if workload["kind"] == "DisaggregatedSet" and workload["name"] == name
            )
    for workload in resolved:
        if workload["kind"] == "DisaggregatedSet":
            workload["currentReplicas"] = workload.get("manifestReplicas")
    kinds = {workload["kind"] for workload in resolved}
    if len(kinds) > 1:
        raise DeploymentScaleError(
            "Manual scaling cannot mix Deployment and DisaggregatedSet workloads.",
            code="deployment_scale_unsupported",
        )
    by_role: dict[str, list[dict[str, Any]]] = {"prefill": [], "decode": []}
    for workload in resolved:
        by_role[workload["role"]].append(workload)
    prefill = by_role["prefill"]
    decode = by_role["decode"]
    if not prefill and not decode:
        raise DeploymentScaleError(
            "No scalable model-server workload was found for this execution.",
            code="deployment_scale_unsupported",
        )
    kind = next(iter(kinds), "Deployment")
    slices = 1
    if kind == "DisaggregatedSet":
        slice_counts = {
            int(workload.get("slices") or 1) for workload in resolved if workload["kind"] == "DisaggregatedSet"
        }
        if len(slice_counts) > 1:
            raise DeploymentScaleError(
                "DisaggregatedSet roles report inconsistent slice counts.",
                code="deployment_scale_unsupported",
            )
        slices = next(iter(slice_counts), 1)
    manual = (execution.provenance or {}).get("manual_scale") or {}
    return {
        "kind": kind,
        "slices": slices,
        "disaggregatedSet": disaggregated_sets[0] if disaggregated_sets else None,
        "externalRoles": sorted(
            {workload["role"] for workload in resolved if workload.get("scalingMode") == "External"}
        ),
        "prefillReplicas": _role_replicas(prefill),
        "decodeReplicas": _role_replicas(decode),
        "workloads": resolved,
        "manual": bool(manual),
    }


async def deployment_scale_plan(execution_id: str) -> dict[str, Any]:
    """Return the current scalable model-server roles for one execution."""
    execution = _store.get_execution(execution_id)
    if execution is None:
        raise DeploymentExecutionNotFoundError(f"deployment execution not found: {execution_id}")
    return await _build_scale_plan(execution)


def _scale_order(changes: dict[str, int], plan: dict[str, Any]) -> list[str]:
    decreasing = any(target < (plan.get(f"{role}Replicas") or 0) for role, target in changes.items())
    return ["decode", "prefill"] if decreasing else ["prefill", "decode"]


async def scale_deployment_execution(
    execution_id: str,
    request: DeploymentScaleRequest,
    *,
    actor: str | None = None,
) -> DeploymentExecutionContext:
    """Scale live model-server Deployments and record the desired replica counts."""
    async with _execution_lock(execution_id):
        execution = _store.get_execution(execution_id)
        if execution is None:
            raise DeploymentExecutionNotFoundError(f"deployment execution not found: {execution_id}")
        plan = await _build_scale_plan(execution)
        workloads = plan["workloads"]
        changes: dict[str, int] = {}
        if request.prefill_replicas is not None:
            if plan.get("prefillReplicas") is None:
                raise DeploymentScaleError(
                    "This deployment has no Prefill workload to scale.",
                    code="deployment_scale_unsupported",
                )
            changes["prefill"] = request.prefill_replicas
        if request.decode_replicas is not None:
            if plan.get("decodeReplicas") is None:
                raise DeploymentScaleError(
                    "This deployment has no Decode workload to scale.",
                    code="deployment_scale_unsupported",
                )
            changes["decode"] = request.decode_replicas
        changes = {role: target for role, target in changes.items() if target != plan.get(f"{role}Replicas")}
        if not changes:
            return get_execution_context(execution_id)
        external_roles = set(plan.get("externalRoles") or [])
        blocked = sorted(set(changes) & external_roles)
        if blocked:
            raise DeploymentScaleError(
                f"Externally managed roles cannot be scaled manually: {', '.join(blocked)}.",
                code="deployment_scale_unsupported",
            )
        cluster_id = _execution_cluster_id(execution)
        successes: dict[str, int] = {}
        failures: list[str] = []
        if plan["kind"] == "DisaggregatedSet":
            name = str(plan.get("disaggregatedSet") or "")
            if not name:
                raise DeploymentScaleError(
                    "DisaggregatedSet name is unavailable.",
                    code="deployment_scale_unsupported",
                )
            operations: list[dict[str, Any]] = []
            for role, target in changes.items():
                role_workloads = [
                    workload
                    for workload in workloads
                    if workload["role"] == role and workload["kind"] == "DisaggregatedSet"
                ]
                for workload in role_workloads:
                    operations.append(
                        {
                            "op": "add",
                            "path": f"/spec/roles/{workload['roleIndex']}/spec/replicas",
                            "value": target,
                        }
                    )
            try:
                result = await run_kubectl(
                    [
                        "patch",
                        "disaggregatedset",
                        name,
                        "--namespace",
                        str(execution.namespace),
                        "--type=json",
                        "--patch",
                        json.dumps(operations, separators=(",", ":")),
                    ],
                    timeout=60,
                    cluster_id=cluster_id,
                )
            except (FileNotFoundError, TimeoutError, OSError) as error:
                failures.append(f"DisaggregatedSet/{name}: {error}")
            else:
                if result.returncode != 0:
                    failures.append(
                        f"DisaggregatedSet/{name}: {(result.stderr or result.stdout or 'scale failed').strip()}"
                    )
                else:
                    successes.update(changes)
        else:
            for role in _scale_order(changes, plan):
                target = changes.get(role)
                if target is None:
                    continue
                role_workloads = [workload for workload in workloads if workload["role"] == role]
                role_failed = False
                for workload in role_workloads:
                    try:
                        result = await run_kubectl(
                            [
                                "scale",
                                "deployment",
                                str(workload["name"]),
                                "--namespace",
                                str(execution.namespace),
                                f"--replicas={target}",
                            ],
                            timeout=60,
                            cluster_id=cluster_id,
                        )
                    except (FileNotFoundError, TimeoutError, OSError) as error:
                        failures.append(f"{role} {workload['name']}: {error}")
                        role_failed = True
                        continue
                    if result.returncode != 0:
                        failures.append(
                            f"{role} {workload['name']}: {(result.stderr or result.stdout or 'scale failed').strip()}"
                        )
                        role_failed = True
                if not role_failed:
                    successes[role] = target
                if failures:
                    break

        provenance = dict(execution.provenance or {})
        manual = dict(provenance.get("manual_scale") or {})
        decode = successes.get("decode", manual.get("decodeReplicas", plan.get("decodeReplicas")))
        prefill = successes.get("prefill", manual.get("prefillReplicas", plan.get("prefillReplicas")))
        if decode is not None:
            manual["decodeReplicas"] = decode
        if prefill is not None:
            manual["prefillReplicas"] = prefill
        manual["kind"] = plan["kind"]
        manual["slices"] = plan["slices"]
        manual["updatedAt"] = utcnow().isoformat()
        manual["updatedBy"] = actor or ""
        provenance["manual_scale"] = manual
        _store.update_execution_provenance(execution_id, provenance)
        if failures:
            raise DeploymentScaleError(
                "Deployment scale was partially applied: " + "; ".join(failures),
                code="deployment_scale_failed",
                status_code=502,
            )
        return get_execution_context(execution_id)


async def update_execution_metadata(
    execution_id: str, request: DeploymentMetadataUpdateRequest
) -> DeploymentExecutionContext:
    """Edit only a deployment's descriptive metadata.

    Runtime facts (model, namespace, image, topology, endpoint, cluster, status)
    are produced by Deploy and stay immutable here.
    """
    async with _execution_lock(execution_id):
        get_execution_context(execution_id)
        _store.update_execution_metadata(execution_id, request)
        return get_execution_context(execution_id)


async def delete_execution(execution_id: str, *, delete_namespace: bool = True) -> None:
    """Remove a deployment's Prism records, optionally cleaning its cluster resources first.

    A deployment still used by another module is never destroyed, and cleanup
    failures keep the records so the delete can be retried. When
    ``delete_namespace`` is false, the Kubernetes namespace and its workloads
    are left running and only Prism's own records are dropped.
    """
    async with _execution_lock(execution_id):
        context = get_execution_context(execution_id)
        reason = active_usage_reason(execution_id)
        if reason:
            raise DeploymentExecutionConflictError(reason, code="deployment_in_use")
        try:
            worker = deployment_run_manager.worker_for_run(context.run_id)
            await worker.delete_case_execution(context.run_id, context.case_id, delete_namespace=delete_namespace)
        except RuntimeConfigurationError as error:
            raise DeploymentExecutionConflictError(
                f"deployment runtime is unavailable, so cluster resources cannot be cleaned: {error}",
                code="deployment_runtime_unavailable",
            ) from error
        except DeploymentInUseError as error:
            raise DeploymentExecutionConflictError(str(error), code="deployment_in_use") from error
        except ValueError as error:
            raise DeploymentExecutionConflictError(str(error)) from error
    _execution_locks.pop(execution_id, None)


async def ensure_execution_endpoint(execution_id: str):
    """Ensure a host-reachable endpoint exists for a deployment execution.

    Reuses a healthy port-forward or rebuilds a dead one, then returns the
    resolved endpoint. Imported lazily so this module stays importable from the
    endpoint resolver itself.
    """
    from llm_d_lens.deploy.endpoint import resolve_deployment_endpoint

    context = get_execution_context(execution_id)
    return await resolve_deployment_endpoint(context.run_id, context.case_id)
