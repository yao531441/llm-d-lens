"""HTTP API for the model service domain.

Three surfaces:
- user (self-scoped): tokens, visible models, own usage
- admin: routing groups/members, usage reports
- internal: Edge Envoy ext_authz authorize + usage recording

Design reference: docs/design/model-service-design.md#6-domain-model
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_d_lens.api.problems import problem
from llm_d_lens.auth.access import (
    current_principal,
    effective_cluster_filter,
    filter_by_cluster,
    owner_for_create,
    require_cluster_access,
    resource_readable,
    visible_cluster_ids,
)
from llm_d_lens.cluster.registry import gateway_lock
from llm_d_lens.deploy.usage import register_usage_probe
from llm_d_lens.model_service.contracts import (
    AuthorizeRequest,
    GroupCreateRequest,
    GroupPublishRequest,
    GroupStatusRequest,
    GroupUpdateRequest,
    MemberCreateRequest,
    MemberUpdateRequest,
    ModelGrantRequest,
    ProvidersStatusRequest,
    TokenCreateRequest,
    TrafficUpdateRequest,
    UsageRecordRequest,
)
from llm_d_lens.model_service.gateway_contracts import (
    ClusterGatewayRequest,
    ComponentLogsRequest,
    ComponentScaleRequest,
    GatewayReconcileRequest,
    GatewayStreamRequest,
    IppConfigRequest,
)
from llm_d_lens.model_service.gateway_ops import (
    GatewayOpsError,
    GatewayOpsService,
    gateway_base_url,
    local_public_ip,
)
from llm_d_lens.model_service.service import (
    ModelServiceConflictError,
    ModelServiceNotFoundError,
    ModelServiceUnauthorizedError,
    default_service,
)
from llm_d_lens.model_service.targets import execution_name
from llm_d_lens.model_service.usage_sync import GatewayUsageSync

logger = logging.getLogger(__name__)
_service = default_service()
_gateway_ops = GatewayOpsService()


def _deployment_usage_reason(execution_id: str) -> str | None:
    """Keep published deployment records and workloads available until unpublished."""
    for member in _service.list_members():
        if member.execution_id == execution_id:
            return f"deployment is published in model service {member.group_id}; unpublish it first"
    return None


register_usage_probe("model-service", _deployment_usage_reason)

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _stream_gateway_operation(run) -> StreamingResponse:
    """Run ``run(on_log)`` in the background and relay its log lines + final
    operation payload as Server-Sent Events (``log``/``complete``/``error``)."""

    async def events():
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def on_log(message: str) -> None:
            await queue.put({"event": "log", "data": {"message": message}})

        async def worker() -> None:
            try:
                operation = await run(on_log)
                await queue.put({"event": "complete", "data": operation.api_payload()})
            except Exception:  # noqa: BLE001 - relay as an SSE error event
                logger.exception("gateway operation stream failed")
                await queue.put({"event": "error", "data": {"message": "Operation failed unexpectedly."}})

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
                if item["event"] in {"complete", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


router = APIRouter(prefix="/api/v1/model-service", tags=["model-service"])
admin_router = APIRouter(prefix="/api/v1/model-service/admin", tags=["model-service-admin"])
internal_router = APIRouter(prefix="/api/v1/internal/model-gateway", tags=["model-service-internal"])
#: OpenAI-compatible public surface served to model tokens (Edge forwards /v1/*).
public_router = APIRouter(tags=["model-service-public"])


def _user_id(request: Request | None) -> str:
    principal = current_principal(request)
    if principal is None or principal.user_id == "system":
        raise HTTPException(status_code=401, detail="authentication required")
    return principal.user_id


def _request_token(request: Request) -> str:
    """Read a model token from ``Authorization: Bearer`` or ``x-api-key``."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-key", "").strip()


async def _reconcile_clusters(cluster_ids: list[str]) -> None:
    """Auto-sync the shared Gateway and model routes after member changes."""
    for cluster_id in sorted(set(cluster_ids)):
        await _gateway_ops.reconcile_cluster(cluster_id)


async def _apply_publication_change(cluster_ids: list[str], *, actor: str | None) -> Response | None:
    try:
        for cluster_id in sorted(set(cluster_ids)):
            operation = await _gateway_ops.reconcile_cluster(cluster_id, actor=actor)
            if operation.status != "succeeded":
                raise GatewayOpsError(operation.message)
    except (GatewayOpsError, OSError, ValueError):
        logger.exception("publication state saved but Gateway reconciliation failed for %s", cluster_ids)
        return problem(
            502,
            "State saved; Gateway reconciliation failed",
            "The requested state was saved, but routing could not be updated. Reload and use Reconcile to retry.",
            "model_publication_apply_failed",
        )
    return None


# --- user: tokens ------------------------------------------------------------


@router.get(
    "/tokens",
    summary="List the caller's active model access tokens (never plaintext).",
    description="List the caller's active model access tokens. Plaintext token values are never returned.",
    operation_id="list_model_service_tokens",
)
async def list_tokens(request: Request) -> dict:
    user_id = _user_id(request)
    return {"items": [token.api_payload() for token in _service.tokens.list_for_user(user_id, include_revoked=False)]}


@router.post("/tokens", status_code=201, summary="Create a model access token; plaintext returned once.")
async def create_token(request: Request, body: TokenCreateRequest) -> dict:
    user_id = _user_id(request)
    token, raw = _service.tokens.create(user_id, name=body.name)
    payload = token.api_payload()
    payload["token"] = raw
    logger.info("event=model_token_created user_id=%s token_id=%s", user_id, token.id)
    return payload


@router.post("/tokens/regenerate", summary="Revoke the active token(s) and issue a new one.")
async def regenerate_token(request: Request, body: TokenCreateRequest | None = None) -> dict:
    user_id = _user_id(request)
    name = body.name if body is not None else "default"
    token, raw = _service.tokens.regenerate(user_id, name=name)
    payload = token.api_payload()
    payload["token"] = raw
    logger.info("event=model_token_regenerated user_id=%s token_id=%s", user_id, token.id)
    return payload


@router.delete(
    "/tokens/{token_id}",
    status_code=204,
    summary="Revoke a model access token.",
    description="Revoke one model access token owned by the caller. Missing or foreign tokens return 404.",
    operation_id="revoke_model_service_token",
)
async def revoke_token(token_id: str, request: Request) -> Response:
    user_id = _user_id(request)
    if not _service.tokens.revoke(user_id, token_id):
        raise HTTPException(status_code=404, detail=f"token not found: {token_id}")
    logger.info("event=model_token_revoked user_id=%s token_id=%s", user_id, token_id)
    return Response(status_code=204)


# --- user: models + usage ----------------------------------------------------


@router.get(
    "/models",
    summary="List the model names the caller may call.",
    description="Return active services with positive-weight online providers. Consumers need inference permission "
    "in the model's cluster and a user/group grant on the model service. Operators retain non-self-scoped inference "
    "access in their binding's cluster scope. Deployment grants do not confer access. "
    "Includes the cluster display name.",
    operation_id="list_model_service_models",
)
async def list_models(request: Request) -> dict:
    user_id = _user_id(request)
    return {"items": _service.list_model_entries_for_user(user_id)}


@router.get(
    "/connection",
    summary="Gateway connection info for clients (per cluster).",
    description=(
        "Return per-cluster Gateway base URLs for clients. Use after publishing a Model Service to give the user "
        "the callable endpoint; deployment readiness alone does not establish this route."
    ),
    operation_id="get_model_service_connection",
)
async def model_service_connection(request: Request) -> dict:
    """Per-cluster Gateway base URLs a client copies into its OpenAI client.

    Each cluster runs its own shared Gateway with its own address, so callers get
    one connection per cluster they can access. ``LENS_MODEL_GATEWAY_PUBLIC_URL``,
    when set, is a single externally reachable override that applies to every cluster.
    """
    global_url = os.environ.get("LENS_MODEL_GATEWAY_PUBLIC_URL", "").strip()
    allowed = visible_cluster_ids(current_principal(request))
    host_header = (request.headers.get("host") or "").split(":")[0]
    public_host = (
        os.environ.get("LENS_PUBLIC_HOST", "").strip()
        or local_public_ip()
        or host_header
        or (request.url.hostname or "")
    )
    status = await _gateway_ops.status()
    clusters: list[dict] = []
    for cluster in status.clusters:
        if allowed is not None and cluster.cluster_id not in allowed:
            continue
        base_url = gateway_base_url(cluster, public_host, global_url)
        clusters.append(
            {
                "clusterId": cluster.cluster_id,
                "clusterName": cluster.cluster_name,
                "provider": cluster.gateway_provider,
                "gatewayName": cluster.gateway_name,
                "address": cluster.gateway_address,
                "publicUrl": cluster.gateway_public_url,
                "port": cluster.gateway_port,
                "baseUrl": base_url,
                "ready": bool(cluster.gateway_ready),
            }
        )
    return {"baseUrl": global_url, "clusters": clusters}


@public_router.get(
    "/v1/models",
    summary="OpenAI-compatible model list for a model token.",
    description="Validate the model token and list callable active services with positive-weight online providers. "
    "Consumers require inference cluster scope plus a model-service user/group grant; non-self-scoped operators "
    "retain inference access in scope. Uses the same entitlement rule as Gateway authorization.",
)
async def public_list_models(request: Request) -> Response:
    """OpenAI ``GET /v1/models`` served by the control plane (decision D13).

    Edge Envoy routes ``/v1/models`` to Python; the token is validated here and
    the response lists only the model names this token may call (the same
    visibility rule as ext_authz).
    """
    token = _request_token(request)
    record = _service.tokens.validate(token, ip=request.client.host if request.client else None)
    if record is None:
        return problem(401, "Invalid token", "token is missing, expired or revoked", "token_invalid")
    data = [
        {
            "id": group.name,
            "object": "model",
            "created": int(group.created_at.timestamp()),
            "owned_by": "lens",
        }
        for group in _service.list_models_for_user(record.user_id)
    ]
    return JSONResponse({"object": "list", "data": data})


@router.get(
    "/usage/records",
    summary="List the caller's recent usage records.",
    description="List the caller's recent model inference usage records, newest first, with limit and offset.",
    operation_id="list_model_service_usage_records",
)
async def usage_records(request: Request, limit: int = 100, offset: int = 0) -> dict:
    user_id = _user_id(request)
    records = _service.usage.list_for_user(user_id, limit=min(max(limit, 1), 500), offset=max(offset, 0))
    return {"items": [record.api_payload() for record in records]}


@router.get(
    "/usage/timeseries",
    summary="Daily token totals for the caller.",
    description="Return daily token totals for the caller over the requested number of days.",
    operation_id="get_model_service_usage_timeseries",
)
async def usage_timeseries(request: Request, days: int = 30) -> dict:
    user_id = _user_id(request)
    since = datetime.now(tz=None) - timedelta(days=max(1, min(days, 365)))
    records = _service.usage.list_for_user(user_id, since=since.replace(tzinfo=None), limit=10000)
    return {"items": _daily_totals(records)}


def _daily_totals(records) -> list[dict]:
    buckets: dict[str, dict] = {}
    for record in records:
        day = record.created_at.date().isoformat()
        bucket = buckets.setdefault(
            day,
            {
                "date": day,
                "requests": 0,
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "cacheWriteTokens": 0,
                "outputTokens": 0,
            },
        )
        bucket["requests"] += 1
        bucket["inputTokens"] += record.input_tokens
        bucket["cachedInputTokens"] += record.cached_input_tokens
        bucket["cacheWriteTokens"] += record.cache_write_tokens
        bucket["outputTokens"] += record.output_tokens
    return [buckets[key] for key in sorted(buckets)]


# --- admin: groups / members / usage -----------------------------------------


@admin_router.get(
    "/groups",
    summary="List model routing groups in the caller's cluster scope.",
    description=(
        "List public Model Services visible in the caller's cluster scope. Use this before publishing to check "
        "whether the desired public model name already exists."
    ),
    operation_id="list_model_service_groups",
)
async def admin_list_groups(request: Request) -> dict:
    principal = current_principal(request)
    groups = [
        group
        for group in _service.list_groups()
        if resource_readable(
            principal,
            permission="model-service:group:read",
            resource_type="cluster",
            resource_id=group.cluster_id or "",
            cluster_id=group.cluster_id,
        )
    ]
    return {"items": [group.api_payload() for group in groups]}


@admin_router.post(
    "/groups",
    status_code=201,
    summary="Create a model service with an optional ready deployment.",
    description="Create a cluster-scoped public model name. With executionId, derive the deployment model names "
    "server-side and create the service and first member in one transaction, then schedule gateway reconciliation. "
    "Combined creation requires group and member management permissions in the deployment's cluster. "
    "Returns the created group (201); inaccessible clusters or missing permissions return 403, "
    "missing deployment targets return 404, and duplicate names, non-ready providers or cluster mismatch return 409. "
    "Without executionId, retains standalone group creation.",
    operation_id="create_model_service_group",
)
async def admin_create_group(request: Request, body: GroupCreateRequest, background: BackgroundTasks) -> dict:
    try:
        principal = current_principal(request)
        group = _service.create_group(body, created_by=owner_for_create(principal), principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model group conflict", str(error), "model_group_conflict")
    if body.execution_id is not None:
        background.add_task(_reconcile_clusters, [group.cluster_id])
    return group.api_payload()


@admin_router.post(
    "/groups/publish",
    status_code=201,
    summary="Publish a ready deployment as a new public model name.",
    description=(
        "Publish a ready deployment as a new public Model Service: a cluster-scoped public model name "
        "plus its first Gateway provider. This is the publishing step, not deployment; deployment "
        "readiness alone does not create a Model Service. Use list_publishable_deployments to choose "
        "the execution, then verify with get_model_service_group. Requires model-service:member:manage "
        "in the deployment's cluster. Returns 409 for duplicate names, non-ready providers or cluster mismatch."
    ),
    operation_id="publish_model_service",
)
async def admin_publish_group(request: Request, body: GroupPublishRequest, background: BackgroundTasks) -> dict:
    try:
        principal = current_principal(request)
        group = _service.publish_group(body, created_by=owner_for_create(principal), principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model group conflict", str(error), "model_group_conflict")
    background.add_task(_reconcile_clusters, [group.cluster_id])
    return group.api_payload()


@admin_router.get(
    "/groups/{group_id}",
    summary="Get a model routing group with cluster-scoped read permission.",
    description=(
        "Return one Model Service's public name, providers and status. Use after publish_model_service or "
        "create_model_service_member to verify that publication and its provider are present."
    ),
    operation_id="get_model_service_group",
)
async def admin_get_group(group_id: str, request: Request) -> dict:
    group = _service.require_group_access(group_id, current_principal(request), "model-service:group:read")
    return group.api_payload()


@admin_router.get(
    "/groups/{group_id}/access",
    summary="List a model service's inference grants.",
    description="Requires model-service:group:share in the model's cluster. Returns user/group grants with display "
    "names, binding IDs and inference permission. A missing model returns 404; insufficient scope returns 403.",
    operation_id="list_model_service_access",
)
async def list_model_access(group_id: str, request: Request) -> dict:
    return {"items": _service.list_access(group_id, principal=current_principal(request))}


@admin_router.get(
    "/groups/{group_id}/access/options",
    summary="List eligible model inference grant recipients.",
    description="Requires scoped model-service:group:share. Returns only user IDs/usernames and group IDs/names "
    "for recipients with inference permission in this model's cluster; does not expose the full identity directory. "
    "Only active, non-protected users are selectable. Model grants do not establish cluster scope.",
    operation_id="list_model_service_access_options",
)
async def model_access_options(group_id: str, request: Request) -> dict:
    return _service.access_options(group_id, principal=current_principal(request))


@admin_router.post(
    "/groups/{group_id}/access",
    status_code=201,
    summary="Grant inference access to a model service.",
    description="Requires model-service:group:share in the model's cluster. Accepts subjectType (user/group) and "
    "subjectId only. The server derives cluster and inference-only grant role; no management, direct deployment "
    "access or onward sharing is granted. Recipient must already have inference cluster scope. Returns the grant "
    "ID and recipient (201), 403 for insufficient scope, 404 for missing resources, or 409 for duplicate grants. "
    "The grant covers all providers of this model without changing HTTPRoute weights.",
    operation_id="grant_model_service_access",
)
async def grant_model_access(group_id: str, body: ModelGrantRequest, request: Request) -> dict:
    return _service.grant_access(group_id, body, principal=current_principal(request))


@admin_router.delete(
    "/groups/{group_id}/access/{binding_id}",
    status_code=204,
    summary="Revoke a model service's inference grant.",
    description="Requires scoped model-service:group:share. Deletes only the specified model's grant and invalidates "
    "recipient authorization. New inference requests re-evaluate access; existing streams are not interrupted. "
    "A missing or mismatched binding returns 404. Other grants and HTTPRoute weights are unchanged.",
    operation_id="revoke_model_service_access",
)
async def revoke_model_access(group_id: str, binding_id: str, request: Request) -> Response:
    _service.revoke_access(group_id, binding_id, principal=current_principal(request))
    return Response(status_code=204)


@admin_router.put(
    "/groups/{group_id}/traffic",
    summary="Save and apply model traffic distribution.",
    description="Atomically update all provider weights and optionally add ready llm-d deployments in the same cluster "
    "with the same provider model name. Weights are integers from 0 to 1000000; zero receives no traffic. "
    "At least one non-disabled provider must have positive weight; at most 16 distinct pools are supported. "
    "expectedWeights must match the previously read executionId-to-weight map. Returns 409 for stale or invalid "
    "allocations, 404 for missing targets, and 403 for insufficient cluster-scoped member-management permission. "
    "After commit, reconciles the cluster Gateway and returns members and the operation result. A failed operation "
    "does not roll back saved weights: configuration remains saved and can be retried with Reconcile. "
    "Known reconciliation exceptions return 502 model_traffic_apply_failed, explicitly indicating saved configuration.",
    operation_id="update_model_service_traffic",
)
async def admin_update_traffic(group_id: str, body: TrafficUpdateRequest, request: Request) -> dict:
    principal = current_principal(request)
    try:
        members = _service.update_traffic(group_id, body, principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Traffic distribution conflict", str(error), "model_traffic_conflict")
    try:
        operation = await _gateway_ops.reconcile_cluster(members[0].cluster_id, actor=principal.user_id)
    except (GatewayOpsError, OSError, ValueError):
        logger.exception("traffic configuration saved but Gateway reconciliation failed for group %s", group_id)
        return problem(
            502,
            "Configuration saved; Gateway reconciliation failed",
            "Traffic weights were saved. Reload the configuration and use Reconcile to retry.",
            "model_traffic_apply_failed",
        )
    return {"items": [member.api_payload() for member in members], "operation": operation.api_payload()}


@admin_router.patch(
    "/groups/{group_id}",
    summary="Update or stop/resume a model routing group.",
    description="Save group configuration and reconcile affected cluster routes before returning. "
    "Status disabled stops this public model; active restores routing to enabled positive-weight providers. "
    "Provider states, weights and running Pods are preserved. A 502 model_publication_apply_failed means "
    "configuration was saved but routing could not be applied; use Reconcile to retry.",
    operation_id="update_model_service_group",
)
async def admin_update_group(group_id: str, body: GroupUpdateRequest, request: Request) -> dict:
    principal = current_principal(request)
    previous = _service.groups.get(group_id)
    cluster_ids = [member.cluster_id for member in _service.list_members(group_id)]
    if previous and previous.cluster_id:
        cluster_ids.append(previous.cluster_id)
    try:
        group = _service.update_group(group_id, body, principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model group conflict", str(error), "model_group_conflict")
    if group.cluster_id:
        cluster_ids.append(group.cluster_id)
    failure = await _apply_publication_change(cluster_ids, actor=principal.user_id if principal else None)
    if failure is not None:
        return failure
    return group.api_payload()


@admin_router.post(
    "/groups/{group_id}/status",
    summary="Enable or disable a model service.",
    description=(
        "Start or stop serving a public model without changing provider states, weights or running Pods. "
        "Requires model-service:group:toggle in the model's cluster; provider and metadata edits keep "
        "their existing permissions."
    ),
    operation_id="set_model_service_status",
)
async def admin_set_group_status(group_id: str, body: GroupStatusRequest, request: Request) -> dict:
    principal = current_principal(request)
    previous = _service.groups.get(group_id)
    cluster_ids = [member.cluster_id for member in _service.list_members(group_id)]
    if previous and previous.cluster_id:
        cluster_ids.append(previous.cluster_id)
    try:
        group = _service.update_group(
            group_id,
            GroupUpdateRequest(status=body.status),
            principal=principal,
            required_permission="model-service:group:toggle",
        )
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model group conflict", str(error), "model_group_conflict")
    if group.cluster_id:
        cluster_ids.append(group.cluster_id)
    failure = await _apply_publication_change(cluster_ids, actor=principal.user_id if principal else None)
    if failure is not None:
        return failure
    return group.api_payload()


@admin_router.delete(
    "/groups/{group_id}",
    status_code=204,
    summary="Safely remove an empty model.",
    description="Requires group management in the model's cluster. Returns 409 model_group_conflict "
    "while any provider remains, including offline providers. Removes the model's Lens-managed HTTPRoute "
    "and IPP mapping and confirms deletion before deleting the record. Returns 404 for a missing model; "
    "502 model_removal_failed preserves the record for retry when cleanup fails. Deployments, Pods, "
    "other models and shared Gateway resources are not deleted. Associated model inference grants are revoked.",
    operation_id="delete_model_service_group",
)
async def admin_delete_group(group_id: str, request: Request) -> Response:
    principal = current_principal(request)
    try:
        group = _service.require_empty_group(
            group_id, principal=principal, required_permission="model-service:group:remove"
        )
        async with gateway_lock(group.cluster_id):
            group = _service.require_empty_group(
                group_id, principal=principal, required_permission="model-service:group:remove"
            )
            await _gateway_ops.remove_model_route(group)
            _service.delete_group(group_id, principal=principal, required_permission="model-service:group:remove")
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model group conflict", str(error), "model_group_conflict")
    except (GatewayOpsError, FileNotFoundError, TimeoutError, OSError, ValueError) as error:
        logger.warning("Model removal cleanup failed for %s: %s", group_id, error)
        return problem(
            502,
            "Model removal failed",
            "Route cleanup could not be confirmed. The model record was retained; retry Remove model.",
            "model_removal_failed",
            expose_detail=True,
        )
    return Response(status_code=204)


@admin_router.put(
    "/groups/{group_id}/providers/status",
    summary="Start or stop all model providers.",
    description="Requires member management in the model's cluster. Atomically sets every provider active "
    "or disabled, preserving weights, model status and Pods, then reconciles routing. Starting requires "
    "every deployment to be ready in its registered cluster. Empty models, unready deployments or concurrent "
    "provider changes return 409 model_member_conflict without partial status updates; a missing model returns "
    "404. A 502 model_publication_apply_failed means statuses were saved but routing failed; use Reconcile.",
    operation_id="set_model_service_providers_status",
)
async def admin_set_providers_status(group_id: str, body: ProvidersStatusRequest, request: Request) -> dict:
    principal = current_principal(request)
    try:
        members = _service.set_providers_status(group_id, body, principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model member conflict", str(error), "model_member_conflict")
    failure = await _apply_publication_change(
        [member.cluster_id for member in members], actor=principal.user_id if principal else None
    )
    if failure is not None:
        return failure
    return {"items": [member.api_payload() for member in members]}


@admin_router.get(
    "/members",
    summary="List published model service members.",
    description="List published model service members, optionally filtered by group, within cluster scope.",
    operation_id="list_model_service_members",
)
async def admin_list_members(request: Request, group_id: str | None = None) -> dict:
    principal = current_principal(request)
    members = filter_by_cluster(_service.list_members(group_id), lambda member: member.cluster_id, principal)
    items = []
    for member in members:
        payload = member.api_payload()
        payload["name"] = execution_name(member.execution_id) or member.execution_id
        items.append(payload)
    return {"items": items}


@admin_router.get(
    "/deployments",
    summary="Ready deployments that can be published as members.",
    description=(
        "List ready llm-d deployments with a deployment-owned InferencePool and EPP that can be published to a "
        "Model Service. Plain vLLM deployments are excluded. This does not publish anything: use "
        "publish_model_service for a new public model name or create_model_service_member to add a provider to an "
        "existing one."
    ),
    operation_id="list_publishable_deployments",
)
async def admin_publishable_deployments(request: Request) -> dict:
    items = _service.list_publishable_deployments(current_principal(request))
    return {"items": [item.api_payload() for item in items]}


@admin_router.post(
    "/members",
    status_code=201,
    summary="Publish a deployment as a group member.",
    description=(
        "Add a ready deployment as a provider to an existing public Model Service, with default weight 1 and "
        "preserving existing provider weights. Use this when the public model name already exists; use "
        "publish_model_service to create a new public name. Schedules Gateway reconciliation; use traffic "
        "distribution to change the allocation."
    ),
    operation_id="create_model_service_member",
)
async def admin_create_member(request: Request, body: MemberCreateRequest, background: BackgroundTasks) -> dict:
    principal = current_principal(request)
    try:
        member = _service.create_member(body, published_by=owner_for_create(principal), principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model member conflict", str(error), "model_member_conflict")
    background.add_task(_reconcile_clusters, [member.cluster_id])
    return member.api_payload()


@admin_router.patch(
    "/members/{member_id}",
    summary="Take a model deployment offline or online.",
    description="Save member status and reconcile its cluster routing before returning. Disabled excludes this "
    "deployment from the model's route without scaling Pods or changing its saved weight. Restoring active "
    "requires a ready deployment; weight 0 remains standby, and a stopped parent model remains stopped. "
    "All providers may be offline. A 409 rejects an unready deployment; a 502 model_publication_apply_failed "
    "means state was saved but routing could not be applied.",
    operation_id="update_model_service_member",
)
async def admin_update_member(member_id: str, body: MemberUpdateRequest, request: Request) -> dict:
    principal = current_principal(request)
    try:
        member = _service.update_member(member_id, body, principal=principal)
    except ModelServiceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceConflictError as error:
        return problem(409, "Model member conflict", str(error), "model_member_conflict")
    failure = await _apply_publication_change([member.cluster_id], actor=principal.user_id if principal else None)
    if failure is not None:
        return failure
    return member.api_payload()


@admin_router.delete(
    "/members/{member_id}",
    status_code=204,
    summary="Remove a member.",
    description="Remove a model service member and reconcile its cluster routing. Missing members return 404.",
    operation_id="delete_model_service_member",
)
async def admin_delete_member(member_id: str, background: BackgroundTasks, request: Request) -> Response:
    member = _service.members.get(member_id)
    _service.delete_member(member_id, principal=current_principal(request))
    if member is not None:
        background.add_task(_reconcile_clusters, [member.cluster_id])
    return Response(status_code=204)


def _parse_dt(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


@admin_router.get(
    "/usage/analytics",
    summary="Aggregated usage analytics (totals, series, per-dimension).",
    description="Return aggregated model usage totals, time series and per-dimension breakdowns with optional filters.",
    operation_id="get_model_service_usage_analytics",
)
async def admin_usage_analytics(
    request: Request,
    since: str | None = None,
    until: str | None = None,
    interval: str = "day",
    group_by: str = "model",
    user_id: Annotated[list[str] | None, Query()] = None,
    group_id: Annotated[list[str] | None, Query()] = None,
    cluster_id: Annotated[list[str] | None, Query()] = None,
) -> dict:
    return _service.usage_analytics(
        since=_parse_dt(since),
        until=_parse_dt(until),
        interval=interval,
        group_by=group_by,
        user_ids=user_id,
        group_ids=group_id,
        cluster_ids=cluster_id,
        principal=current_principal(request),
    )


@admin_router.get(
    "/usage",
    summary="Recent usage records across users/models/clusters.",
    description="List recent model usage records across visible users, models and clusters, "
    "with optional group and cluster filters.",
    operation_id="list_model_service_usage",
)
async def admin_usage(
    request: Request,
    group_id: Annotated[list[str] | None, Query()] = None,
    cluster_id: Annotated[list[str] | None, Query()] = None,
    limit: int = 100,
) -> dict:
    effective_cluster_ids = effective_cluster_filter(current_principal(request), cluster_id)
    records = _service.usage.list_all(
        group_ids=group_id, cluster_ids=effective_cluster_ids, limit=min(max(limit, 1), 500)
    )
    return {"items": [record.api_payload() for record in records]}


# --- admin: gateway operations ----------------------------------------------


@admin_router.get(
    "/gateway/status",
    summary="Gateway data-plane status across clusters.",
    description="Return shared Gateway, IPP, route and last-operation status for clusters visible to the caller.",
    operation_id="get_model_gateway_status",
)
async def admin_gateway_status(request: Request) -> dict:
    status = await _gateway_ops.status()
    status.clusters = filter_by_cluster(status.clusters, lambda cluster: cluster.cluster_id, current_principal(request))
    return status.api_payload()


@admin_router.post(
    "/gateway/reconcile",
    summary="Sync the shared Gateway and model routes for every cluster with members.",
    description="Schedule Gateway and HTTPRoute reconciliation. Each enabled positive-weight provider contributes "
    "a weighted InferencePool backendRef; zero-weight providers remain stored but are omitted. Returns the scheduled "
    "cluster IDs, not proof of controller acceptance or observed traffic ratios.",
    operation_id="reconcile_model_gateway",
)
async def admin_gateway_reconcile(background: BackgroundTasks) -> dict:
    cluster_ids = sorted({member.cluster_id for member in _service.members.list_all() if member.status == "active"})
    background.add_task(_reconcile_clusters, cluster_ids)
    return {"scheduled": True, "clusters": cluster_ids}


@admin_router.post(
    "/gateway/probe",
    summary="Probe members' backends now and refresh health/status.",
    description="Refresh non-disabled providers of enabled models and return the updated count. Stopped models "
    "and offline providers are skipped, and health writes cannot re-enable an offline provider. A zero-weight "
    "InferencePool without Gateway parent status is standby, not degraded; that alone does not trigger "
    "Gateway/IPP auto-healing. Missing resources and endpoint failures still mark the provider unhealthy.",
    operation_id="probe_model_gateway",
)
async def admin_gateway_probe() -> dict:
    updated = await _gateway_ops.probe_members()
    return {"updated": updated}


@admin_router.get(
    "/gateway/ipp-config",
    summary="Read a cluster's live Inference Payload Processor config.",
    description="Read the live Inference Payload Processor config for one accessible cluster.",
    operation_id="get_model_gateway_ipp_config",
)
async def admin_gateway_get_ipp_config(request: Request, cluster_id: str = Query(alias="clusterId")) -> dict:
    require_cluster_access(current_principal(request), cluster_id)
    return await _gateway_ops.get_ipp_config(cluster_id)


@admin_router.post(
    "/gateway/ipp-config",
    summary="Set a cluster's Inference Payload Processor config and reinstall it.",
    description="Replace a cluster's Inference Payload Processor config and reinstall the IPP.",
    operation_id="set_model_gateway_ipp_config",
)
async def admin_gateway_ipp_config(body: IppConfigRequest, request: Request) -> dict:
    require_cluster_access(current_principal(request), body.cluster_id)
    operation = await _gateway_ops.set_ipp_config(
        body.cluster_id, body.config, actor=owner_for_create(current_principal(request))
    )
    return operation.api_payload()


@admin_router.post(
    "/gateway/usage/sync",
    summary="Scrape EPP token counters and record usage deltas (per user/API key).",
    description="Scrape EPP token counters and record usage deltas for all visible model services.",
    operation_id="sync_model_gateway_usage",
)
async def admin_gateway_usage_sync() -> dict:
    sync = GatewayUsageSync(service=_service)
    written = await sync.sync()
    return {"written": written}


@admin_router.post(
    "/gateway/preflight",
    summary="Preflight a cluster (kubectl reachable, active members).",
    description="Check kubectl reachability and active members before installing a shared Gateway in a cluster.",
    operation_id="preflight_model_gateway",
)
async def admin_gateway_preflight(body: ClusterGatewayRequest, request: Request) -> dict:
    require_cluster_access(current_principal(request), body.cluster_id)
    operation = await _gateway_ops.preflight(body.cluster_id, actor=owner_for_create(current_principal(request)))
    return operation.api_payload()


@admin_router.post(
    "/gateway/install",
    summary="Install or reuse the cluster's shared llm-d Gateway.",
    description="Install in a namespace suffixed with the stable Lens cluster-registration short ID. "
    "Gateway, IPP and authentication resources are isolated from other registrations. Existing unsuffixed "
    "resources are not adopted or deleted. A NodePort still used by another Gateway is reported as a conflict.",
    operation_id="install_model_gateway",
)
async def admin_gateway_install(body: ClusterGatewayRequest, request: Request) -> dict:
    require_cluster_access(current_principal(request), body.cluster_id)
    operation = await _gateway_ops.install_cluster_gateway(
        body.cluster_id, provider=body.provider, actor=owner_for_create(current_principal(request))
    )
    return operation.api_payload()


@admin_router.post(
    "/gateway/scale",
    summary="Scale a data-plane component Deployment (Gateway data plane / EPP / model server).",
    description="Scale a Gateway data-plane, EPP or model-server Deployment to the requested replica count.",
    operation_id="scale_model_gateway",
)
async def admin_gateway_scale(body: ComponentScaleRequest, request: Request) -> dict:
    require_cluster_access(current_principal(request), body.cluster_id)
    operation = await _gateway_ops.scale_component(
        body.cluster_id,
        body.namespace,
        body.name,
        body.replicas,
        kind=body.kind,
        actor=owner_for_create(current_principal(request)),
    )
    return operation.api_payload()


@admin_router.post("/gateway/stream", summary="Run a data-plane operation and stream its log lines (SSE).")
async def admin_gateway_stream(body: GatewayStreamRequest, request: Request) -> StreamingResponse:
    principal = current_principal(request)
    require_cluster_access(principal, body.cluster_id)
    actor = owner_for_create(principal)

    async def run(on_log):
        if body.op == "install-gateway":
            return await _gateway_ops.install_cluster_gateway(
                body.cluster_id, provider=body.provider, actor=actor, on_log=on_log
            )
        if body.op == "uninstall-gateway":
            return await _gateway_ops.uninstall_inference_gateway(body.cluster_id, actor=actor)
        if body.op == "reconcile":
            return await _gateway_ops.reconcile_cluster_gateway(body.cluster_id, actor=actor, on_log=on_log)
        if body.op == "scale":
            if not body.namespace or not body.name or body.replicas is None:
                raise ValueError("scale requires namespace, name and replicas")
            return await _gateway_ops.scale_component(
                body.cluster_id, body.namespace, body.name, body.replicas, kind=body.kind, actor=actor
            )
        if body.op in ("gateway-start", "gateway-stop"):
            return await _gateway_ops.scale_gateway(
                body.cluster_id, 0 if body.op == "gateway-stop" else 1, actor=actor, on_log=on_log
            )
        if body.op == "install-ipp":
            return await _gateway_ops.install_ipp(
                body.cluster_id,
                namespace=body.namespace,
                name=body.name or "lens-ipp",
                provider=body.provider,
                gateway_name=body.gateway_name,
                chart=body.ipp_chart,
                version=body.ipp_version,
                actor=actor,
                on_log=on_log,
            )
        if body.op == "uninstall-ipp":
            return await _gateway_ops.uninstall_ipp(
                body.cluster_id,
                namespace=body.namespace,
                name=body.name or "lens-ipp",
                actor=actor,
                on_log=on_log,
            )
        raise ValueError(f"unknown operation: {body.op}")

    return _stream_gateway_operation(run)


@admin_router.post(
    "/gateway/reconcile/stream",
    summary="Reconcile a cluster's shared Gateway and model routes (SSE).",
    description=(
        "Re-render and reapply the shared Gateway, HTTPRoutes and provider wiring for one cluster. "
        "This is an idempotent maintenance action available to scoped maintainers."
    ),
)
async def admin_gateway_reconcile_cluster(body: GatewayReconcileRequest, request: Request) -> StreamingResponse:
    principal = current_principal(request)
    require_cluster_access(principal, body.cluster_id)
    actor = owner_for_create(principal)

    async def run(on_log):
        return await _gateway_ops.reconcile_cluster_gateway(body.cluster_id, actor=actor, on_log=on_log)

    return _stream_gateway_operation(run)


@admin_router.post("/gateway/logs", summary="Stream a data-plane component's logs (SSE).")
async def admin_gateway_logs(body: ComponentLogsRequest, request: Request) -> StreamingResponse:
    require_cluster_access(current_principal(request), body.cluster_id)

    async def events():
        try:
            namespace, name = await _gateway_ops.resolve_log_target(
                body.cluster_id, component=body.component, namespace=body.namespace, name=body.name
            )
            async for line in _gateway_ops.stream_component_logs(
                body.cluster_id,
                namespace,
                name,
                kind=body.kind,
                container=body.container,
                tail=body.tail,
            ):
                yield f"event: log\ndata: {json.dumps({'message': line})}\n\n"
        except Exception:  # noqa: BLE001 - relay as an SSE error event
            logger.exception("Unable to stream model service logs")
            yield f"event: error\ndata: {json.dumps({'message': 'Unable to stream model service logs'})}\n\n"
        yield "event: complete\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# --- internal: ext_authz + usage ---------------------------------------------


@internal_router.post(
    "/authorize",
    summary="ext_authz: validate token and authorize model access.",
    description="Validate the inference token and authorize the model in the requested cluster. Consumers require "
    "inference cluster scope and a direct or group model grant. Non-self-scoped operators retain inference in scope. "
    "An active positive-weight provider is required. Gateway independently selects its weighted pool; "
    "the legacy member is not actual "
    "Gateway destination attribution. Invalid tokens return 401; unavailable or forbidden models return 404.",
)
async def internal_authorize(request: Request) -> Response:
    return await _authorize(request, None)


@internal_router.api_route(
    "/authorize",
    methods=["GET", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
async def internal_authorize_other(request: Request) -> Response:
    return await _authorize(request, None)


@internal_router.api_route(
    "/authorize/{rest:path}",
    methods=["POST", "PUT", "PATCH", "DELETE", "GET"],
    include_in_schema=False,
)
async def internal_authorize_rest(request: Request, rest: str) -> Response:
    return await _authorize(request, rest)


async def _authorize(request: Request, rest: str | None) -> Response:
    """Envoy ext_authz entry point.

    Envoy forwards the client's HTTP method to the authz service, so every method
    is accepted here. Accepts either a JSON body (``{token, model}``) or the raw
    OpenAI request: the token is read from ``Authorization`` / ``x-api-key`` and
    the model from the body (or ``x-lens-model``). On success returns 200 with
    routing headers (``x-lens-*`` and ``x-ai-eg-model``) that Edge Envoy forwards
    upstream. ``/v1/models`` has no model and is only token-checked; Edge routes
    it to the control plane.
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - body may be empty/non-JSON
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    authorization = request.headers.get("authorization", "")
    header_token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    token = str(payload.get("token") or header_token or request.headers.get("x-api-key", "") or "")
    model = str(
        payload.get("model")
        or request.headers.get("x-lens-model", "")
        or request.headers.get("x-gateway-base-model-name", "")
        or ""
    )
    target = (rest or "").strip("/")
    # The Gateway's ext_authz path carries its cluster id (".../authorize/cluster/<id>/...").
    cluster_match = re.search(r"(?:^|/)cluster/([A-Za-z0-9_-]+)", target)
    cluster_id = cluster_match.group(1) if cluster_match else None
    if target.endswith("v1/models"):
        record = _service.tokens.validate(token)
        if record is None:
            return problem(
                401,
                "Invalid token",
                "token is missing, expired or revoked",
                "token_invalid",
                headers={"x-lens-error": "invalid token"},
            )
        return JSONResponse(
            {"object": "list"},
            headers={"x-lens-user-id": hashlib.sha256(record.user_id.encode()).hexdigest()[:32]},
        )
    if not token or not model:
        return problem(
            401,
            "Invalid token",
            "missing token or model",
            "token_invalid",
            headers={"x-lens-error": "missing token or model"},
        )
    client_ip = request.client.host if request.client else None
    try:
        result = _service.authorize_request(
            AuthorizeRequest(token=token, model=model, client_ip=client_ip), cluster_id=cluster_id
        )
    except ModelServiceUnauthorizedError as error:
        return problem(401, "Invalid token", str(error), "token_invalid", headers={"x-lens-error": "invalid token"})
    except ModelServiceNotFoundError as error:
        return problem(
            404, "Model not found", str(error), "model_not_found", headers={"x-lens-error": "model not found"}
        )
    headers = {
        "x-lens-user-id": hashlib.sha256(result.user_id.encode()).hexdigest()[:32],
        "x-lens-token": result.token_id,
        "x-lens-group": result.group_id,
        "x-lens-group-name": result.group_name,
        "x-lens-model-ref": result.model_ref,
        "x-lens-cluster": result.cluster_id,
        "x-lens-member": result.execution_id,
        "x-lens-model": result.model_ref or model,
        "x-ai-eg-model": model,
        # llm-d EPP reads this as the per-request `fairness_id` metric label, which
        # the usage sync maps back to this token/user (see usage_sync.py).
        "x-llm-d-inference-fairness-id": result.token_id,
    }
    return JSONResponse(content=result.model_dump(mode="json", by_alias=True), headers=headers)


@internal_router.post("/record-usage", summary="Record a completed inference's token usage (idempotent).")
async def internal_record_usage(body: UsageRecordRequest) -> dict:
    record = _service.record_usage(body)
    return {"recorded": True, "id": record.id}


@internal_router.get("/models", summary="Model names a specific user may call (for /v1/models).")
async def internal_models(user_id: str) -> dict:
    return {"items": [group.api_payload() for group in _service.list_models_for_user(user_id)]}
