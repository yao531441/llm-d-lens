"""HTTP API for local Configuration-driven deployment runs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from llm_d_lens.api.problems import problem
from llm_d_lens.auth.access import (
    cluster_allowed,
    current_principal,
    owner_for_create,
    require_cluster_access,
    resource_readable,
    visible_cluster_ids,
)
from llm_d_lens.db.dao.deployment_batch import DeploymentBatchDao
from llm_d_lens.deploy.application import deployment_run_manager
from llm_d_lens.deploy.contracts import (
    DeploymentCaseStatus,
    DeploymentMetadataUpdateRequest,
    DeploymentRunCreateRequest,
    DeploymentRunStatus,
    DeploymentScaleRequest,
    DeploymentStatus,
)
from llm_d_lens.deploy.endpoint import DeploymentEndpointResolutionError
from llm_d_lens.deploy.executions import (
    DeploymentExecutionConflictError,
    DeploymentExecutionNotFoundError,
    DeploymentScaleError,
    ensure_execution_endpoint,
    execution_deployment_plan,
    find_execution_context,
    list_execution_contexts,
    scale_deployment_execution,
    serving_metadata,
    update_execution_metadata,
)
from llm_d_lens.deploy.executions import (
    delete_execution as delete_execution_record,
)
from llm_d_lens.deploy.orphans import (
    OrphanCleanupError,
    clean_orphan_namespaces,
    namespace_prefix,
    scan_orphan_namespaces,
)
from llm_d_lens.deploy.providers.hardware_profile import active_profile, discover_runtime_classes
from llm_d_lens.deploy.providers.target_node import validate_placement
from llm_d_lens.deploy.standard_kubernetes_service import (
    StandardKubernetesServiceRequest,
    build_standard_kubernetes_service_configuration,
)
from llm_d_lens.deploy.usage import DeploymentInUseError
from llm_d_lens.hardware.resolver import deployment_mode
from llm_d_lens.infrastructure.kubernetes import PortForwardError, list_resources

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])
_store = deployment_run_manager.store
_refresh_task: asyncio.Task[None] | None = None
logger = logging.getLogger(__name__)


class CleanCaseRequest(BaseModel):
    preserve_rendered_overlay: bool = False


class RebindClusterSessionRequest(BaseModel):
    cluster_session_id: str


class CleanOrphansRequest(BaseModel):
    cluster_id: str = Field(alias="clusterId")
    namespaces: list[str] = Field(default_factory=list)


class RestartCaseRequest(BaseModel):
    model_token: str | None = None


class RuntimeVerificationRequest(BaseModel):
    smoke: bool = Field(
        default=False,
        description=(
            "Send one bounded inference request, collect version-supported request-correlated NIXL "
            "evidence and metrics, and persist the report; may establish a port-forward."
        ),
    )
    metrics_only: bool = Field(
        default=False,
        description=(
            "Read current model-server metric counters for owned Pods without sending any inference "
            "request or persisting a report. Mutually exclusive with smoke."
        ),
    )


@router.post(
    "/plans",
    operation_id="prepare_deployment_plan",
    summary="Freeze and preflight a deployment plan",
    description=(
        "Validate target-bound configurations and runtime settings, render final workloads and Router inputs, "
        "and persist an owner-bound immutable plan. Downloads a pinned Chart but creates no Run or Kubernetes "
        "resources. Returns digest, reviewable manifests and pass/fail/unknown preflight evidence. "
        "Invalid or unsupported configurations return 422."
    ),
)
async def prepare_deployment_plan(request: DeploymentRunCreateRequest, http_request: Request):
    from llm_d_lens.deploy.prepared_plans import prepare, public_plan

    require_cluster_access(current_principal(http_request), request.provenance.get("cluster_server_id"))
    try:
        return public_plan(
            await prepare(
                deployment_run_manager, request, owner_user_id=owner_for_create(current_principal(http_request))
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get(
    "/plans/{plan_id}",
    operation_id="get_deployment_plan",
    summary="Read a persisted deployment plan",
    description=(
        "Return a plan's digest, frozen manifests, Router values and preflight evidence. Requires deployment "
        "read permission, cluster access and ownership; unavailable plans return 404. Chart archive bytes and "
        "local paths are not returned. Reading a plan does not refresh its recorded preflight."
    ),
)
async def get_deployment_plan(plan_id: str, http_request: Request):
    from llm_d_lens.db.dao.deployment_plan import DeploymentPlanDao
    from llm_d_lens.deploy.prepared_plans import public_plan

    plan = DeploymentPlanDao().get(plan_id)
    principal = current_principal(http_request)
    if plan is None or plan.owner_user_id != owner_for_create(principal):
        raise HTTPException(status_code=404, detail="deployment plan not found")
    require_cluster_access(principal, plan.cluster_id)
    return public_plan(plan)


@router.get(
    "/topology",
    operation_id="get_deployment_topology",
    summary="Read raw cluster topology inventory",
    description=(
        "Return Node labels, allocatable resources, scheduling status and effective Pod requests from the selected "
        "cluster. This is inventory, not plan-specific eligibility or a capacity reservation. Cluster access and "
        "deployment-create permission are required. Incomplete or unavailable reads return 503, never an empty success."
    ),
)
async def get_deployment_topology(cluster_id: str, http_request: Request):
    from llm_d_lens.configuration.disaggregated_set import resource_requests
    from llm_d_lens.deploy.prepared_plans import inventory

    require_cluster_access(current_principal(http_request), cluster_id)
    try:
        nodes = await inventory(cluster_id, "nodes")
        pods = await inventory(cluster_id, "pods")
        return {
            "clusterId": cluster_id,
            "nodes": [
                {
                    "name": item["metadata"]["name"],
                    "labels": item["metadata"].get("labels", {}),
                    "allocatable": item.get("status", {}).get("allocatable", {}),
                    "conditions": item.get("status", {}).get("conditions", []),
                    "taints": item.get("spec", {}).get("taints", []),
                    "unschedulable": item.get("spec", {}).get("unschedulable", False),
                }
                for item in nodes
            ],
            "pods": [
                {
                    "namespace": item["metadata"].get("namespace"),
                    "name": item["metadata"]["name"],
                    "node": item.get("spec", {}).get("nodeName"),
                    "phase": item.get("status", {}).get("phase"),
                    "resourceRequests": resource_requests(item.get("spec", {})),
                }
                for item in pods
            ],
        }
    except (ValueError, TimeoutError, OSError) as error:
        raise HTTPException(status_code=503, detail="Cluster topology inventory unavailable") from error


@router.post(
    "/executions/{execution_id}/verify",
    operation_id="verify_deployment_execution",
    summary="Verify actual DS placement and runtime evidence",
    description=(
        "Read DS, LWS, Pod ownership, observed generations and actual Node domains for a prepared execution. "
        "Returns independent Kubernetes, placement, routing and KV-transfer evidence. Optional smoke sends one "
        "bounded inference request, may establish a port-forward and records metric deltas plus vLLM 0.26.0 NIXL pull "
        "completion/producer acknowledgements. HTTP success does not imply KV transfer. Smoke reports are persisted "
        "as deployment artifacts; passive reads include lastProbe when available. Optional metrics_only reads "
        "current per-Pod metric counters with a collection timestamp and sends no request. Requires case-execute "
        "scope and cluster access. Unsupported executions return 409; "
        "unavailable cluster evidence returns 503."
    ),
)
async def verify_deployment_execution(execution_id: str, body: RuntimeVerificationRequest, http_request: Request):
    import httpx

    from llm_d_lens.db.dao.deployment_plan import DeploymentPlanDao
    from llm_d_lens.deploy.prepared_plans import plan_documents
    from llm_d_lens.deploy.runtime_verification import verify_execution

    if body.smoke and body.metrics_only:
        raise HTTPException(status_code=422, detail="smoke and metrics_only are mutually exclusive")
    context = _require_execution(execution_id)
    _require_execution_permission(current_principal(http_request), context, "deployment:case:execute")
    require_cluster_access(current_principal(http_request), context.cluster_id)
    run = _store.get_run(context.run_id) if context.run_id else None
    plan = DeploymentPlanDao().get(run.provenance.get("_prepared_plan_id")) if run else None
    if plan is None:
        raise HTTPException(status_code=409, detail="Execution has no prepared DS plan")
    index = next((index for index, case in enumerate(run.cases) if case.id == context.case_id), None)
    if index is None:
        raise HTTPException(status_code=409, detail="Prepared execution case is unavailable")
    try:
        return await verify_execution(
            context, plan_documents(plan, index), smoke=body.smoke, metrics_only=body.metrics_only
        )
    except (httpx.HTTPError, KeyError, IndexError) as error:
        raise HTTPException(status_code=503, detail="Runtime inference evidence unavailable") from error
    except (ValueError, TimeoutError, OSError) as error:
        raise HTTPException(status_code=503, detail="Runtime evidence is unavailable") from error


def _ready_model_cache_entry(entries, model: str):
    expected = model.strip().lower()
    for entry in entries:
        source = getattr(entry, "source", None)
        huggingface = getattr(source, "huggingface", None)
        repo_id = str(getattr(huggingface, "repo_id", "") or "").strip().lower()
        if repo_id == expected and getattr(entry, "status", "") == "ready":
            return entry
    return None


def _ready_model_cache_entry_exists(entries, model: str) -> bool:
    return _ready_model_cache_entry(entries, model) is not None


async def _refresh_deployment_record(run, *, pending_only: bool = False) -> None:
    try:
        refreshable_cases = {
            DeploymentCaseStatus.DEPLOYING,
            DeploymentCaseStatus.READY,
            DeploymentCaseStatus.FAILED,
        }
        if not any(
            case.execution_id and case.status in refreshable_cases for case in run.cases
        ) or not deployment_run_manager.has_available_source(run):
            return
        worker = deployment_run_manager.worker_for_run(run.id)
        await worker.refresh_run_executions(run.id, pending_only=pending_only)
    except Exception:
        logger.exception("Unable to refresh deployment run %s", run.id)


async def _refresh_all_deployment_records(*, pending_only: bool = False) -> None:
    # FAILED belongs here for the same reason FAILED is a refreshable case
    # above: a rollout that exceeded its progress deadline is recorded as
    # failed, but the workload often recovers afterwards (a CrashLoopBackOff
    # that clears once its model cache is complete, for example). Without the
    # run itself being refreshable, that recovery is never observed and the
    # deployment stays red forever even though its Pods are serving.
    # ``refresh_diagnostics`` promotes such an execution back to READY, and a
    # namespace that really is gone is corrected to CLEANED instead.
    refreshable_statuses = {
        DeploymentRunStatus.RUNNING,
        DeploymentRunStatus.SUCCEEDED,
        DeploymentRunStatus.PARTIALLY_SUCCEEDED,
        DeploymentRunStatus.FAILED,
    }
    await asyncio.gather(
        *(
            _refresh_deployment_record(run, pending_only=pending_only)
            for run in _store.list_runs()
            if run.status in refreshable_statuses
            and (not pending_only or any(case.status == DeploymentCaseStatus.DEPLOYING for case in run.cases))
        )
    )


async def _refresh_deployment_records() -> None:
    next_full_refresh = 0.0
    while True:
        now = asyncio.get_running_loop().time()
        full_refresh = now >= next_full_refresh
        await _refresh_all_deployment_records(pending_only=not full_refresh)
        if full_refresh:
            next_full_refresh = now + 300
        # Pending deployments need prompt readiness/monitoring reconciliation.
        # Ready deployments retain the slower, more expensive diagnostic refresh.
        await asyncio.sleep(5)


@router.on_event("startup")
async def start_deployment_record_refresh() -> None:
    global _refresh_task
    for run in _store.list_runs():
        if run.status == DeploymentRunStatus.CANCELLING:
            deployment_run_manager.reconcile_interrupted_cancellation(run.id)
        elif run.status in {DeploymentRunStatus.QUEUED, DeploymentRunStatus.RUNNING}:
            deployment_run_manager.resume_run(run.id)
    if _refresh_task is None or _refresh_task.done():
        _refresh_task = asyncio.create_task(_refresh_deployment_records())


@router.on_event("shutdown")
async def stop_deployment_record_refresh() -> None:
    global _refresh_task
    if _refresh_task is not None:
        _refresh_task.cancel()
        await asyncio.gather(_refresh_task, return_exceptions=True)
        _refresh_task = None


async def _validate_cluster_placement(placement: dict | None, cluster_id: str | None, http_request: Request) -> None:
    if placement is None:
        return
    validate_placement(placement)
    node_label = placement.get("nodeLabel")
    if node_label is not None:
        if not cluster_id:
            raise ValueError("cluster_server_id is required for a node label")
        require_cluster_access(current_principal(http_request), cluster_id)
        nodes = await list_resources("nodes", cluster_id=cluster_id)
        if not any(
            (node.get("metadata") or {}).get("labels", {}).get(node_label["key"]) == node_label["value"]
            for node in nodes
        ):
            raise ValueError("selected node label is not present in the cluster")


@router.post(
    "/runs",
    summary="Start a deployment run from one or more already-rendered deployable configurations.",
    description=(
        "Start a deployment run from one or more already-rendered deployable configurations. "
        "DisaggregatedSet configurations require prepared_plan_id and expected_plan_digest from a reviewed plan; "
        "unknown preflight constraints require acknowledge_unknown=true. Changed inputs or failed rechecks return 409. "
        "Use this when you want Lens Deploy to execute a configuration artifact or custom deployable payload. "
        "The run starts queued and its cases may show pods stuck Pending if the cluster currently lacks free "
        "resources to schedule them -- tell the user right away that it looks resource-constrained and queued, "
        "rather than silently waiting for it to finish."
    ),
    operation_id="create_deployment_run",
)
async def create_run(request: DeploymentRunCreateRequest, http_request: Request = None):
    try:
        placement = request.runtime_binding.placement if request.runtime_binding else None
        await _validate_cluster_placement(placement, request.provenance.get("cluster_server_id"), http_request)
        return await deployment_run_manager.start_run(
            request, owner_user_id=owner_for_create(current_principal(http_request))
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get(
    "/node-labels",
    summary="List available node labels for deployment placement.",
    description="Return distinct node labels and values from the selected accessible cluster so callers can "
    "choose a placement label for a deployment.",
    operation_id="list_deployment_node_labels",
)
async def list_deployment_node_labels(cluster_id: str = Query(..., min_length=1), request: Request = None) -> dict:
    require_cluster_access(current_principal(request), cluster_id)
    nodes = await list_resources("nodes", cluster_id=cluster_id)
    labels = {
        (key, value)
        for node in nodes
        for key, value in ((node.get("metadata") or {}).get("labels") or {}).items()
        if isinstance(key, str) and isinstance(value, str)
    }
    return {"items": [{"key": key, "value": value} for key, value in sorted(labels)]}


@router.get(
    "/runtime-classes",
    summary="Discover RuntimeClasses available to a target deployment cluster.",
    description=(
        "List RuntimeClass names, handlers, scheduling constraints and matching nodes for an accessible cluster. "
        "An optional hardware profile and access mode identify a profile suggestion, not a runtime requirement. "
        "Discovery failures are reported, never treated as an empty list. Class existence cannot verify node "
        "handler, CDI or driver readiness. Select auto, node-default or named through runtime_binding.runtime_class."
    ),
    operation_id="list_deployment_runtime_classes",
)
async def list_deployment_runtime_classes(
    cluster_id: str = Query(..., min_length=1),
    accelerator: str | None = None,
    access_mode: str | None = None,
    request: Request = None,
) -> dict:
    require_cluster_access(current_principal(request), cluster_id)
    profile = active_profile(accelerator) if accelerator else None
    if access_mode and profile is None:
        raise HTTPException(status_code=422, detail="An access mode requires a registered hardware profile")
    if accelerator and profile is None:
        raise HTTPException(status_code=422, detail="Unknown hardware profile")
    try:
        mode = deployment_mode(profile, access_mode) if profile else {}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        capabilities = await discover_runtime_classes(cluster_id)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    items = []
    for runtime in capabilities["items"]:
        scheduling = runtime.get("scheduling") or {}
        selector = scheduling.get("nodeSelector") or {}
        items.append(
            {
                "name": runtime["metadata"]["name"],
                "handler": runtime.get("handler"),
                "scheduling": scheduling,
                "overhead": runtime.get("overhead") or {},
                "matching_nodes": [
                    node["metadata"]["name"]
                    for node in capabilities["nodes"]
                    if all(
                        node.get("metadata", {}).get("labels", {}).get(key) == value for key, value in selector.items()
                    )
                ],
            }
        )
    suggested = mode.get("runtime_class_name")
    return {
        "items": items,
        "profile_suggestion": suggested,
        "available_suggestion": suggested if any(item["name"] == suggested for item in items) else None,
        "requirement": "unknown",
        "node_runtime_readiness": "unknown",
        "warnings": [
            "Class existence does not verify node handler, CDI or driver readiness. "
            "Use node-default only when the node default runtime provides device access."
        ],
    }


@router.post(
    "/standard-vllm-runs",
    summary="Start a Standard Kubernetes Service deployment run using Lens\\'s allowlisted vLLM contract.",
    description=(
        "Start a Standard Kubernetes Service deployment run using Lens\\'s allowlisted vLLM contract. "
        "Use this for direct baseline or simple model-service deployments without a prebuilt configuration artifact."
    ),
    operation_id="create_standard_vllm_deployment_run",
)
async def create_standard_kubernetes_service_run(
    request: StandardKubernetesServiceRequest, http_request: Request = None
):
    """Create the current vLLM implementation of a Standard service run."""
    try:
        from llm_d_lens.cluster import registry
        from llm_d_lens.cluster import service as cluster_service
        from llm_d_lens.cluster.sessions import require_active_session
        from llm_d_lens.deploy.providers.hardware_profile import cluster_accelerator

        session = require_active_session(request.cluster_session_id)
        await _validate_cluster_placement(
            {
                "replicas": request.placement_replicas,
                **({"nodeLabel": request.placement_node_label} if request.placement_node_label else {}),
            },
            session.server_id if session else None,
            http_request,
        )
        model_revision = None
        if request.storage_type == "model-cache":
            from llm_d_lens.model_cache.service import default_service as model_cache_service
            from llm_d_lens.storage.contracts import StorageVolumeKind, StorageVolumePurpose
            from llm_d_lens.storage.service import get_ready_volume

            volume = await get_ready_volume(request.storage_volume_id, cluster_id=session.server_id)
            if StorageVolumePurpose.MODEL_CACHE not in volume.purposes:
                raise ValueError("selected storage volume is not designated for model cache use")
            if volume.kind == StorageVolumeKind.DYNAMIC_PVC:
                raise ValueError("model-cache storage for deployments cannot use dynamic-pvc volumes")
            entries = await model_cache_service().list(
                cluster_id=session.server_id,
                storage_volume_id=request.storage_volume_id,
            )
            ready_entry = _ready_model_cache_entry(entries, request.model)
            if ready_entry is None:
                raise ValueError("selected model is not ready in the selected model cache storage")
            model_revision = ready_entry.source.huggingface.revision
        overview = await cluster_service.build_overview(registry.require_cluster(session.server_id))
        accelerator = cluster_accelerator(overview.get("hardware") or {})
        configuration = build_standard_kubernetes_service_configuration(
            request, accelerator=accelerator, model_revision=model_revision
        )
        return await deployment_run_manager.start_run(
            DeploymentRunCreateRequest(
                configurations=[configuration],
                provenance={
                    "deployment_source": {"kind": "standard-kubernetes-service"},
                    **({"cluster_server_id": session.server_id} if session else {}),
                    "cluster_session_id": request.cluster_session_id,
                    "deployment_name": request.deployment_name,
                    "description": request.description.strip(),
                    "model_market": {
                        "deployment_name": request.deployment_name,
                        "description": request.description.strip(),
                    },
                },
                runtime_binding={
                    "runtime_class": request.runtime_class,
                    "placement": {
                        "replicas": request.placement_replicas,
                        **({"nodeLabel": request.placement_node_label} if request.placement_node_label else {}),
                    },
                },
            ),
            owner_user_id=owner_for_create(current_principal(http_request)),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _owner_for_run(run_id: str | None):
    if not run_id:
        return None
    return DeploymentBatchDao().owner_map([run_id]).get(run_id)


def _run_readable(principal, run_id: str, cluster_id, owner) -> bool:
    owner_user_id, owner_group_id = owner if owner else (None, None)
    return resource_readable(
        principal,
        permission="deployment:run:read",
        resource_type="deployment_run",
        resource_id=run_id,
        cluster_id=cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    )


def _execution_readable(principal, context, owner) -> bool:
    """Evaluate the run or concrete execution's management scope and ownership."""
    owner_user_id, owner_group_id = owner if owner else (None, None)
    if context.run_id and resource_readable(
        principal,
        permission="deployment:run:read",
        resource_type="deployment_run",
        resource_id=context.run_id,
        cluster_id=context.cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    ):
        return True
    return resource_readable(
        principal,
        permission="deployment:run:read",
        resource_type="deployment_execution",
        resource_id=context.execution_id,
        cluster_id=context.cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    )


def _require_run_permission(principal, run, permission: str) -> None:
    if principal is None:
        return
    owner = _owner_for_run(run.id)
    owner_user_id, owner_group_id = owner if owner else (None, None)
    if not resource_readable(
        principal,
        permission=permission,
        resource_type="deployment_run",
        resource_id=run.id,
        cluster_id=run.provenance.get("cluster_server_id"),
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    ):
        raise HTTPException(status_code=403, detail="deployment run is not accessible")


def _require_execution_permission(principal, context, permission: str) -> None:
    if principal is None:
        return
    owner = _owner_for_run(context.run_id)
    owner_user_id, owner_group_id = owner if owner else (None, None)
    if context.run_id and resource_readable(
        principal,
        permission=permission,
        resource_type="deployment_run",
        resource_id=context.run_id,
        cluster_id=context.cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    ):
        return
    if resource_readable(
        principal,
        permission=permission,
        resource_type="deployment_execution",
        resource_id=context.execution_id,
        cluster_id=context.cluster_id,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
    ):
        return
    raise HTTPException(status_code=403, detail="deployment execution is not accessible")


def _require_execution(execution_id: str):
    context = find_execution_context(execution_id)
    if context is None:
        raise HTTPException(status_code=404, detail="deployment execution not found")
    return context


def _execution_payload(context, principal, owner=None):
    """Expose canonical links and existing policy decisions without exposing owner identities."""
    if owner is None:
        owner = _owner_for_run(context.run_id)
    user_id, group_id = owner if owner else (None, None)
    ownership = "unknown" if not user_id else "self" if principal and user_id == principal.user_id else "other"

    def allowed(permission, *, run_only=False):
        if principal is None:
            return False
        return any(
            resource_readable(
                principal,
                permission=permission,
                resource_type=kind,
                resource_id=identifier,
                cluster_id=context.cluster_id,
                owner_user_id=user_id,
                owner_group_id=group_id,
            )
            for kind, identifier in (("deployment_run", context.run_id), ("deployment_execution", context.execution_id))
            if identifier and (not run_only or kind == "deployment_run")
        )

    payload = context.api_payload()
    payload.update(
        run_id=context.run_id,
        case_id=context.case_id,
        ownership=ownership,
        capabilities={
            "update": allowed("deployment:execution:update"),
            "stop": allowed("deployment:case:execute", run_only=True),
            "delete": allowed("deployment:execution:delete"),
        },
    )
    return payload


def _list_runs(query: str = "", principal=None):
    needle = query.lower().strip()
    runs = list(reversed(_store.list_runs()))
    if principal is not None:
        owners = DeploymentBatchDao().owner_map([run.id for run in runs])
        runs = [
            run
            for run in runs
            if _run_readable(principal, run.id, run.provenance.get("cluster_server_id"), owners.get(run.id))
        ]
    if not needle:
        return runs
    return [
        run
        for run in runs
        if needle in run.id.lower()
        or any(
            needle in case.provider_ref.lower()
            or needle in str(case.create_request.deployment_policy.value.get("deployment_name", "")).lower()
            or needle
            in str(
                _store.get_execution(case.execution_id).namespace
                if case.execution_id and _store.get_execution(case.execution_id)
                else ""
            ).lower()
            for case in run.cases
        )
    ]


@router.get(
    "",
    summary="List deployment runs from the collection root.",
    description=(
        "Return the same deployment run collection as the /runs endpoint. Use the optional query parameter to "
        "search by run id, provider ref, deployment name, or namespace."
    ),
    operation_id="list_deployments",
)
async def list_deployments(request: Request = None, query: str = ""):
    """List deployment runs for clients that use the collection root."""
    return _list_runs(query, current_principal(request))


@router.post(
    "/executions/{execution_id}/restart-plan",
    operation_id="prepare_deployment_execution_restart",
    summary="Prepare a reviewed deployment restart",
    description=(
        "Prepare a new immutable plan for a stopped or failed prepared execution. Returns status=awaiting_review, "
        "plan and request for digest-bound POST /runs confirmation. Does not clean resources or schedule execution. "
        "Requires run-create permission, case-execute scope and cluster access. Optional cluster_session_id renews "
        "the same-cluster session with session-rebind permission. Evaluation-owned executions require workflow retry. "
        "Unsupported executions return 409."
    ),
)
async def prepare_execution_restart(
    execution_id: str, http_request: Request, request: RebindClusterSessionRequest | None = None
):
    context = _require_execution(execution_id)
    _require_execution_permission(current_principal(http_request), context, "deployment:case:execute")
    run = _store.get_run(context.run_id)
    if run is None or not run.provenance.get("_prepared_plan_id"):
        raise HTTPException(status_code=409, detail="Execution has no prepared deployment plan")
    if request is not None:
        await rebind_cluster_session(context.run_id, request, http_request)
    return await restart_case(context.run_id, context.case_id, RestartCaseRequest(), http_request)


@router.get(
    "/runs",
    summary="List deployment runs Lens has recorded.",
    description=(
        "List deployment runs Lens has recorded. Use the optional query string to search by run id, provider ref, "
        "deployment name, or namespace."
    ),
    operation_id="list_deployment_runs",
)
async def list_runs(request: Request = None, query: str = ""):
    return _list_runs(query, current_principal(request))


@router.get(
    "/executions",
    summary="List deployment executions without creating new port-forwards.",
    description=(
        "List deployment executions without creating new port-forwards. Use this to discover execution ids, "
        "readiness, cluster affinity, and endpoint metadata. Missing serving fields are supplemented from stored "
        "rendered vLLM manifests; flat runtime parameters describe decode (or aggregated serving), not prefill. "
        "Unspecified or unparseable parameters remain unknown."
    ),
    operation_id="list_deployment_executions",
)
async def list_executions(
    request: Request = None,
    status: DeploymentStatus | None = None,
    statuses: Annotated[list[DeploymentStatus] | None, Query()] = None,
    cluster_id: str = "",
    query: str = "",
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
):
    """List deployment executions without creating endpoint port-forwards."""
    principal = current_principal(request)
    contexts = list_execution_contexts(
        status=status,
        statuses=statuses,
        cluster_id=cluster_id,
        query=query,
        limit=limit,
        allowed_cluster_ids=visible_cluster_ids(principal),
    )
    owners = DeploymentBatchDao().owner_map([context.run_id for context in contexts])
    if principal is not None:
        contexts = [
            context for context in contexts if _execution_readable(principal, context, owners.get(context.run_id))
        ]
    return {"items": [_execution_payload(context, principal, owners.get(context.run_id)) for context in contexts]}


@router.get(
    "/executions/{execution_id}",
    summary="Get one deployment execution by execution id.",
    description=(
        "Get one deployment execution by execution id. Use this to inspect the current deployment status, namespace, "
        "provenance, and endpoint metadata. Missing serving fields are supplemented from stored rendered vLLM "
        "manifests; flat runtime parameters describe decode (or aggregated serving), not prefill. "
        "Unspecified or unparseable parameters remain unknown."
    ),
    operation_id="get_deployment_execution",
)
async def get_execution(execution_id: str, request: Request = None):
    """Return one deployment execution keyed by its own id."""
    context = find_execution_context(execution_id)
    if context is None:
        raise HTTPException(status_code=404, detail="deployment execution not found")
    principal = current_principal(request)
    _require_execution_permission(principal, context, "deployment:run:read")
    payload = _execution_payload(context, principal)
    run = _store.get_run(context.run_id) if context.run_id else None
    case = next((case for case in run.cases if case.id == context.case_id), None) if run else None
    if case and 0 <= case.source_configuration_ordinal < len(run.source_configurations):
        payload["configuration"] = run.source_configurations[case.source_configuration_ordinal].model_dump(mode="json")
    return payload


@router.get(
    "/executions/{execution_id}/plan",
    summary="Get the deployment plan and rendered Kubernetes YAML of one execution.",
    description=(
        "Get the deployment plan and rendered Kubernetes YAML of one execution. Returns the configuration "
        "artifacts the deployment was created from (`configuration_artifacts`, each with its `kind`, "
        "`provider_ref` and inline `content`) and the rendered multi-document manifest that Lens applied "
        "(`manifest`), so the plan can be audited or re-applied with `kubectl apply -f`. `manifest` is null while "
        "the execution is still pending or when the rendered file is no longer stored or cannot be parsed. Secret "
        "`data` and `stringData` values, including Secrets in `List.items` and manifests embedded in artifact "
        "content, are replaced with `<redacted>`; restore real values or remove those Secret objects before "
        "re-applying the manifest. Returns 404 when the execution does not exist."
    ),
    operation_id="get_deployment_execution_plan",
)
async def get_execution_plan(execution_id: str, request: Request = None):
    """Return the stored configuration plan and rendered manifest of one execution."""
    context = find_execution_context(execution_id)
    if context is None:
        raise HTTPException(status_code=404, detail="deployment execution not found")
    _require_execution_permission(current_principal(request), context, "deployment:run:read")
    return execution_deployment_plan(context)


def _resolve_execution_cluster_id(context) -> str | None:
    if context.cluster_id:
        return context.cluster_id
    if context.cluster_session_id:
        from llm_d_lens.cluster import sessions

        session = sessions.get_session(context.cluster_session_id)
        if session:
            return session.server_id
    return None


@router.get(
    "/executions/{execution_id}/pods",
    summary="Get live pod status for all pods in a deployment execution\\'s namespace.",
    description=(
        "Get live pod status for all pods in a deployment execution\\'s namespace. Use this for Kubernetes-level "
        "rollout debugging. A pod phase of Pending usually means the cluster does not yet have free resources to "
        "schedule it -- report this to the user immediately as resource-constrained/queued instead of waiting for it "
        "to become Running."
    ),
    operation_id="get_deployment_execution_pods",
)
async def get_execution_pods(execution_id: str, request: Request = None):
    """Return status and metadata for all pods running in the deployment's namespace."""
    context = find_execution_context(execution_id)
    if context is None:
        return {"execution_id": execution_id, "namespace": None, "pods": [], "error": "Deployment initializing..."}
    _require_execution_permission(current_principal(request), context, "deployment:run:read")
    if not context.namespace:
        return {"execution_id": execution_id, "namespace": None, "pods": []}

    cluster_id = _resolve_execution_cluster_id(context)
    from llm_d_lens.infrastructure.kubernetes import run_kubectl

    try:
        result = await run_kubectl(
            ["get", "pods", "--namespace", context.namespace, "-o", "json"],
            cluster_id=cluster_id,
            timeout=15,
        )
    except (FileNotFoundError, TimeoutError) as error:
        logger.warning("Unable to list deployment pods for %s: %s", execution_id, error)
        return {
            "execution_id": execution_id,
            "namespace": context.namespace,
            "pods": [],
            "error": "Unable to list deployment pods",
        }

    if result.returncode != 0:
        return {
            "execution_id": execution_id,
            "namespace": context.namespace,
            "pods": [],
            "error": (result.stderr or result.stdout or "kubectl get pods failed").strip(),
        }

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"execution_id": execution_id, "namespace": context.namespace, "pods": [], "error": "invalid JSON"}

    items = payload.get("items") or []
    pods = []
    for item in items:
        metadata = item.get("metadata") or {}
        status = item.get("status") or {}
        spec = item.get("spec") or {}
        conditions = status.get("conditions") or []
        container_statuses = status.get("containerStatuses") or status.get("container_statuses") or []
        init_statuses = status.get("initContainerStatuses") or []

        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)

        restarts = sum(cs.get("restartCount", 0) for cs in container_statuses) + sum(
            cs.get("restartCount", 0) for cs in init_statuses
        )

        status_reason = status.get("phase", "Unknown")
        for cs in container_statuses + init_statuses:
            state = cs.get("state") or {}
            if "waiting" in state:
                status_reason = state["waiting"].get("reason") or status_reason
            elif "terminated" in state and state["terminated"].get("reason"):
                status_reason = state["terminated"].get("reason")

        containers = []
        for cs in container_statuses:
            c_state = (
                "running"
                if "running" in cs.get("state", {})
                else "waiting"
                if "waiting" in cs.get("state", {})
                else "terminated"
                if "terminated" in cs.get("state", {})
                else "unknown"
            )
            containers.append(
                {
                    "name": cs.get("name"),
                    "image": cs.get("image"),
                    "ready": bool(cs.get("ready")),
                    "restart_count": cs.get("restartCount", 0),
                    "state": c_state,
                }
            )

        pods.append(
            {
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "labels": metadata.get("labels") or {},
                "phase": status.get("phase", "Unknown"),
                "ready": ready,
                "pod_ip": status.get("podIP"),
                "node_name": spec.get("nodeName"),
                "restarts": restarts,
                "status_reason": status_reason,
                "containers": containers,
                "creation_timestamp": metadata.get("creationTimestamp"),
            }
        )

    return {
        "execution_id": execution_id,
        "namespace": context.namespace,
        "pods": pods,
    }


@router.get(
    "/executions/{execution_id}/pods/{pod_name}/logs",
    summary="Get logs for one pod inside a deployment execution\\'s namespace.",
    description=(
        "Get logs for one pod inside a deployment execution\\'s namespace. Optionally scope to one container, tail "
        "length, or previous container instance."
    ),
    operation_id="get_deployment_pod_logs",
)
async def get_execution_pod_logs(
    execution_id: str,
    pod_name: str,
    request: Request = None,
    container: str | None = None,
    tail: int = Query(default=200, ge=1, le=2000),
    previous: bool = False,
):
    """Return logs for a specific pod or container within the deployment's namespace."""
    context = find_execution_context(execution_id)
    if context is None:
        raise HTTPException(status_code=404, detail="deployment execution not found")
    _require_execution_permission(current_principal(request), context, "deployment:run:read")
    if not context.namespace:
        raise HTTPException(status_code=422, detail="deployment has no namespace")

    cluster_id = _resolve_execution_cluster_id(context)
    from llm_d_lens.infrastructure.kubernetes import run_kubectl

    cmd = ["logs", pod_name, "--namespace", context.namespace, f"--tail={tail}"]
    if container:
        cmd.append(f"--container={container}")
    if previous:
        cmd.append("--previous")

    try:
        result = await run_kubectl(cmd, cluster_id=cluster_id, timeout=20)
    except (FileNotFoundError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    logs = (result.stdout if result.returncode == 0 else result.stderr or result.stdout or "").strip()
    return {
        "execution_id": execution_id,
        "namespace": context.namespace,
        "pod_name": pod_name,
        "container": container,
        "logs": logs,
        "success": result.returncode == 0,
    }


@router.patch(
    "/executions/{execution_id}",
    summary=(
        "Update only the user-owned display metadata on a deployment execution, such as display name or description."
    ),
    description=(
        "Update only the user-owned display metadata on a deployment execution, such as display name or description. "
        "Runtime and provenance fields remain immutable."
    ),
    operation_id="patch_deployment_execution",
)
async def patch_execution(execution_id: str, request: DeploymentMetadataUpdateRequest, http_request: Request = None):
    """Update a deployment's descriptive metadata; runtime facts stay immutable."""
    existing = find_execution_context(execution_id)
    if existing is not None:
        _require_execution_permission(current_principal(http_request), existing, "deployment:execution:update")
    try:
        context = await update_execution_metadata(execution_id, request)
    except DeploymentExecutionNotFoundError as error:
        return problem(404, "Deployment not found", str(error), "deployment_not_found")
    logger.info(
        "event=deployment_metadata_updated execution_id=%s fields=%s",
        execution_id,
        ",".join(
            name
            for name, value in (
                ("display_name", request.display_name),
                ("description", request.description),
            )
            if value is not None
        ),
    )
    return context.api_payload()


@router.post(
    "/executions/{execution_id}/scale",
    summary="Manually scale a deployment's live model-server replicas.",
    description=(
        "Scale the live Prefill and/or Decode model-server Deployments for a ready or failed deployment. "
        "This changes Kubernetes replicas only; the saved configuration is unchanged."
    ),
    operation_id="scale_deployment_execution",
)
async def scale_execution(
    execution_id: str,
    request: DeploymentScaleRequest,
    http_request: Request = None,
):
    """Scale live model-server Deployments without re-rendering the configuration."""
    principal = current_principal(http_request)
    existing = find_execution_context(execution_id)
    if existing is not None:
        _require_execution_permission(principal, existing, "deployment:execution:scale")
    try:
        context = await scale_deployment_execution(
            execution_id,
            request,
            actor=principal.user_id if principal else None,
        )
    except DeploymentExecutionNotFoundError as error:
        return problem(404, "Deployment not found", str(error), "deployment_not_found")
    except DeploymentScaleError as error:
        return problem(error.status_code, "Deployment scale failed", str(error), error.code)
    logger.info(
        "event=deployment_scaled execution_id=%s prefill_replicas=%s decode_replicas=%s",
        execution_id,
        request.prefill_replicas,
        request.decode_replicas,
    )
    return context.api_payload()


@router.delete(
    "/executions/{execution_id}",
    status_code=204,
    summary=("Delete a deployment execution record and optionally clean up its Kubernetes namespace."),
    description=(
        "Delete a deployment execution record and optionally clean up its Kubernetes namespace. "
        "Use this for definitive deployment teardown and record removal."
    ),
    operation_id="delete_deployment_execution",
)
async def delete_execution(execution_id: str, delete_namespace: bool = True, request: Request = None):
    """Remove a deployment's Prism records, optionally cleaning its cluster resources."""
    existing = find_execution_context(execution_id)
    if existing is not None:
        _require_execution_permission(current_principal(request), existing, "deployment:execution:delete")
    logger.info(
        "event=deployment_delete_started execution_id=%s delete_namespace=%s",
        execution_id,
        delete_namespace,
    )
    try:
        await delete_execution_record(execution_id, delete_namespace=delete_namespace)
    except DeploymentExecutionNotFoundError as error:
        return problem(404, "Deployment not found", str(error), "deployment_not_found")
    except DeploymentExecutionConflictError as error:
        logger.info("event=deployment_delete_blocked execution_id=%s code=%s", execution_id, error.code)
        return problem(409, "Deployment cannot be deleted", str(error), error.code)
    except PortForwardError as error:
        logger.warning("event=deployment_delete_cleanup_failed execution_id=%s error=%s", execution_id, error)
        return problem(502, "Deployment cleanup failed", str(error), "deployment_cleanup_failed")
    logger.info("event=deployment_deleted execution_id=%s", execution_id)
    return Response(status_code=204)


@router.post(
    "/executions/{execution_id}/endpoint",
    summary="Create or reuse a reachable forwarded endpoint for a deployment execution and return the local URL.",
    description=(
        "Create or reuse a reachable forwarded endpoint for a deployment execution and return the local URL. "
        "Use this before calling the model from Playground, Simulation, or evaluation tools."
    ),
    operation_id="connect_deployment_execution_endpoint",
)
async def connect_execution_endpoint(execution_id: str, request: Request = None):
    """Port-forward a deployment execution's endpoint and return a reachable URL."""
    existing = find_execution_context(execution_id)
    if existing is not None:
        _require_execution_permission(current_principal(request), existing, "deployment:execution:connect")
    try:
        resolved = await ensure_execution_endpoint(execution_id)
    except DeploymentExecutionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DeploymentEndpointResolutionError as error:
        status_code = 422 if "missing namespace" in str(error) else 404
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PortForwardError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "success": True,
        "execution_id": execution_id,
        "id": resolved.forward_id,
        "local_port": resolved.local_port,
        "endpoint": resolved.endpoint,
        "forwarded_endpoint": resolved.endpoint,
        "namespace": resolved.namespace,
        "service": resolved.service,
        "remote_port": resolved.remote_port,
    }


@router.get(
    "/cluster/{cluster_id}",
    summary="List deployment case records targeting a specific cluster id.",
    description=(
        "List deployment case records targeting a specific cluster id. Use this when you want a cluster-scoped "
        "deployment history rather than execution-centric results."
    ),
    operation_id="list_cluster_deployments",
)
async def list_cluster_deployments(cluster_id: str, request: Request = None):
    """List deployment records (cases) targeting a specific cluster."""
    principal = current_principal(request)
    if principal is not None and not cluster_allowed(principal, cluster_id):
        raise HTTPException(status_code=403, detail="cluster is not accessible")
    deployments: list[dict] = []
    for run in reversed(_store.list_runs()):
        if str(run.provenance.get("cluster_server_id") or "") != cluster_id:
            continue
        if principal is not None and not _run_readable(principal, run.id, cluster_id, _owner_for_run(run.id)):
            continue
        for case in run.cases:
            execution = _store.get_execution(case.execution_id) if case.execution_id else None
            deployments.append(_deployment_payload(case, execution))
    return deployments


@router.get(
    "/orphans",
    summary="List deployment namespaces on a cluster that no Lens record references.",
    description="List namespaces under the deployment prefix with no execution record -- "
    "leftovers from records deleted without namespace cleanup, or from a rebuilt database. "
    "Namespaces holding a Lens engine instance are excluded; the scan fails if an instance check fails.",
    operation_id="list_orphan_deployment_namespaces",
)
async def list_orphan_namespaces(cluster_id: str, request: Request = None) -> dict:
    """List namespaces under the deployment prefix with no Lens execution record."""
    principal = current_principal(request)
    require_cluster_access(principal, cluster_id)
    try:
        orphans = await scan_orphan_namespaces(cluster_id)
    except OrphanCleanupError as error:
        return problem(502, "Orphan scan failed", str(error), "orphan_scan_failed")
    return {
        "clusterId": cluster_id,
        "prefix": namespace_prefix(),
        "items": [orphan.api_payload() for orphan in orphans],
    }


@router.post(
    "/orphans/clean",
    summary="Delete selected orphan deployment namespaces.",
    description="Delete the given orphan namespaces. Names outside the deployment prefix, or "
    "still referenced by a Lens record, are refused rather than deleted. Namespaces holding a Lens engine "
    "instance, or whose instance check fails, are also refused.",
    operation_id="clean_orphan_deployment_namespaces",
)
async def clean_orphan_namespaces_route(body: CleanOrphansRequest, request: Request = None) -> dict:
    """Delete the caller-selected orphan namespaces."""
    principal = current_principal(request)
    require_cluster_access(principal, body.cluster_id)
    logger.info(
        "event=deployment_orphan_clean cluster_id=%s count=%s",
        body.cluster_id,
        len(body.namespaces),
    )
    cleaned, failed = await clean_orphan_namespaces(body.cluster_id, body.namespaces)
    return {"clusterId": body.cluster_id, "cleaned": cleaned, "failed": failed}


@router.get(
    "/runs/{run_id}",
    summary="Get the full detail of one deployment run, including all cases and their current status.",
    description=(
        "Get the full detail of one deployment run, including all cases and their current status. A run or case "
        "status of queued (or pending pods reported by get_deployment_execution_pods) usually means the cluster "
        "does not yet have free resources -- tell the user it looks resource-constrained and queued right away "
        "instead of waiting for it to finish."
    ),
    operation_id="get_deployment_run",
)
async def get_run(run_id: str, request: Request = None):
    run = _store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="deployment run not found")
    _require_run_permission(current_principal(request), run, "deployment:run:read")
    return run


@router.delete(
    "/runs/{run_id}",
    status_code=204,
    summary="Delete a completed or otherwise deletable deployment run record.",
    description=(
        "Delete a completed or otherwise deletable deployment run record. "
        "Use this when the run history should be removed from Lens."
    ),
    operation_id="delete_deployment_run",
)
async def delete_run(run_id: str, request: Request = None):
    run = _store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="deployment run not found")
    _require_run_permission(current_principal(request), run, "deployment:execution:delete")
    try:
        await deployment_run_manager.worker_for_run(run_id).delete_run(run_id)
    except DeploymentInUseError as error:
        return problem(409, "Deployment is in use", str(error), "deployment_in_use")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/runs/{run_id}/cancel",
    status_code=202,
    summary="Request cancellation of an active deployment run.",
    description=(
        "Request cancellation of an active deployment run. Use this to stop queued or running deployment "
        "orchestration work."
    ),
    operation_id="cancel_deployment_run",
)
async def cancel_run(run_id: str, request: Request = None) -> dict[str, str]:
    run = _store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="deployment run not found")
    _require_run_permission(current_principal(request), run, "deployment:run:cancel")
    if run.status.value not in {"queued", "running", "cancelling"}:
        raise HTTPException(status_code=409, detail="deployment run is not active")
    deployment_run_manager.worker_for_run(run_id).request_cancel(run_id)
    return {"id": run_id, "status": "cancelling"}


@router.post(
    "/runs/{run_id}/cluster-session",
    status_code=204,
    summary="Rebind a deployment run to a different active cluster session.",
    description=(
        "Rebind a deployment run to a different active cluster session. Use this after reconnecting to the same "
        "cluster so refresh or cleanup actions can continue."
    ),
    operation_id="rebind_deployment_run_cluster_session",
)
async def rebind_cluster_session(
    run_id: str, request: RebindClusterSessionRequest, http_request: Request = None
) -> None:
    run = _store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="deployment run not found")
    _require_run_permission(current_principal(http_request), run, "deployment:session:rebind")
    try:
        deployment_run_manager.rebind_cluster_session(run_id, request.cluster_session_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get(
    "/runs/{run_id}/cases/{case_id}",
    summary=(
        "Get one deployment case within a deployment run, including its associated execution and deployment summary "
        "payloads."
    ),
    description=(
        "Get one deployment case within a deployment run, including its associated execution and deployment summary "
        "payloads."
    ),
    operation_id="get_deployment_case",
)
async def get_case(run_id: str, case_id: str, request: Request = None):
    run = _store.get_run(run_id)
    if run is not None:
        _require_run_permission(current_principal(request), run, "deployment:run:read")
    case = _store.get_case(run_id, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="deployment case not found")
    execution = _store.get_execution(case.execution_id) if case.execution_id else None
    return _case_response(case, execution)


@router.post(
    "/runs/{run_id}/cases/{case_id}/refresh",
    summary="Refresh a deployment case\\'s execution record from the live cluster.",
    description=(
        "Refresh a deployment case\\'s execution record from the live cluster. Use this when rollout state or "
        "endpoint details may have changed and the stored record is stale."
    ),
    operation_id="refresh_deployment_case",
)
async def refresh_case(run_id: str, case_id: str, request: Request = None):
    run = _store.get_run(run_id)
    if run is not None:
        _require_run_permission(current_principal(request), run, "deployment:case:execute")
    try:
        execution = await deployment_run_manager.worker_for_run(run_id).refresh_case_execution(run_id, case_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return _case_response(_store.get_case(run_id, case_id), execution)


@router.get(
    "/runs/{run_id}/cases/{case_id}/logs",
    summary="Get orchestration logs for one deployment case.",
    description=(
        "Get orchestration logs for one deployment case. Use this to inspect deploy-time actions, status changes, "
        "and provider output recorded by Lens Deploy."
    ),
    operation_id="get_deployment_case_logs",
)
async def get_case_logs(run_id: str, case_id: str, limit: int = 200, request: Request = None):
    run = _store.get_run(run_id)
    if run is not None:
        _require_run_permission(current_principal(request), run, "deployment:run:read")
    try:
        return await deployment_run_manager.worker_for_run(run_id).get_case_logs(
            run_id, case_id, limit=min(max(limit, 1), 1000)
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post(
    "/runs/{run_id}/cases/{case_id}/stop",
    summary="Stop a deployment case and its associated execution.",
    description=(
        "Stop a deployment case and its associated execution. Use this when a single case within a run must be "
        "halted without deleting the whole run."
    ),
    operation_id="stop_deployment_case",
)
async def stop_case(run_id: str, case_id: str, request: Request = None):
    run = _store.get_run(run_id)
    if run is not None:
        _require_run_permission(current_principal(request), run, "deployment:case:execute")
    try:
        case, execution = await deployment_run_manager.worker_for_run(run_id).stop_case(run_id, case_id)
    except DeploymentInUseError as error:
        return problem(409, "Deployment is in use", str(error), "deployment_in_use")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _case_response(case, execution)


@router.post(
    "/runs/{run_id}/cases/{case_id}/restart",
    summary="Restart a deployment case.",
    description=(
        "Restart a deployment case. Optionally supply a model token override when the case needs a fresh secret or "
        "different runtime credential during restart. Prepared DS cases instead return status=awaiting_review, "
        "plan and request, without cleanup or execution; submit request with the reviewed plan ID/digest "
        "to POST /runs. "
        "Prepared retries use the configured Secret reference and reject inline token overrides. "
        "Evaluation-owned DS cases must use workflow retry. Old resources are not removed during preparation."
    ),
    operation_id="restart_deployment_case",
)
async def restart_case(run_id: str, case_id: str, request: RestartCaseRequest, http_request: Request = None):
    run = _store.get_run(run_id)
    if run is not None:
        _require_run_permission(current_principal(http_request), run, "deployment:case:execute")
    try:
        if run is not None and run.provenance.get("_prepared_plan_id"):
            from llm_d_lens.deploy.prepared_plans import prepare, public_plan, restart_request

            if request.model_token:
                raise ValueError(
                    "Prepared retry requires rotating the configured Secret reference, not an inline token"
                )
            principal = current_principal(http_request)
            renewed = restart_request(run, case_id)
            require_cluster_access(principal, renewed.provenance.get("cluster_server_id"))
            plan = await prepare(deployment_run_manager, renewed, owner_user_id=owner_for_create(principal))
            return {"status": "awaiting_review", "plan": public_plan(plan), "request": renewed.model_dump(mode="json")}
        case = await deployment_run_manager.worker_for_run(run_id).restart_case(
            run_id, case_id, model_token=(request.model_token or "").strip() or None
        )
    except DeploymentInUseError as error:
        return problem(409, "Deployment is in use", str(error), "deployment_in_use")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"case": case}


@router.post(
    "/runs/{run_id}/cases/{case_id}/clean",
    summary="Clean up one deployment case\\'s runtime resources and marks it cleaned.",
    description=(
        "Clean up one deployment case\\'s runtime resources and marks it cleaned. Use preserve_rendered_overlay "
        "when you want to retain rendered overlays for later inspection."
    ),
    operation_id="clean_deployment_case",
)
async def clean_case(run_id: str, case_id: str, request: CleanCaseRequest, http_request: Request = None):
    run = _store.get_run(run_id)
    if run is not None:
        _require_run_permission(current_principal(http_request), run, "deployment:case:execute")
    try:
        case, execution = await deployment_run_manager.worker_for_run(run_id).clean_case(
            run_id, case_id, preserve_rendered_overlay=request.preserve_rendered_overlay
        )
    except DeploymentInUseError as error:
        return problem(409, "Deployment is in use", str(error), "deployment_in_use")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _case_response(case, execution)


def _deployment_payload(case, execution):
    serving = serving_metadata(case, execution)
    return {
        "run_id": case.run_id if case else None,
        "case_id": case.id if case else None,
        "execution_id": case.execution_id if case else None,
        "name": serving["name"],
        "display_name": execution.metadata.display_name if execution else "",
        "description": execution.metadata.description if execution else "",
        "status": case.status.value if case else None,
        "namespace": execution.namespace if execution else None,
        "endpoint": execution.endpoint.url if execution and execution.endpoint else None,
        "forwarded_endpoint": execution.forwarded_endpoint if execution else None,
        "guide": serving["guide"],
        "backend": serving["backend"],
        "model_name": serving["model_name"],
        "replicas": serving["replicas"],
        "tensor_parallel_size": serving["tensor_parallel_size"],
        "image": serving["image"],
    }


def _case_response(case, execution):
    return {
        "case": case,
        "execution": execution,
        "deployment": _deployment_payload(case, execution),
    }
