# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Minimal API application for Prism backend workflows."""

import asyncio
import contextlib
import logging
import os
from datetime import UTC, datetime

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from llm_d_lens.agentic.router import router as agentic_deployment_router
from llm_d_lens.ai_providers.router import router as ai_providers_router
from llm_d_lens.aic import router as aic_router
from llm_d_lens.api.problems import install_problem_handlers
from llm_d_lens.api.speech_models import router as speech_models_router
from llm_d_lens.api.web import WebEntryMiddleware
from llm_d_lens.api.web import router as web_router
from llm_d_lens.auth.bootstrap import ensure_initial_admin, format_initial_admin_banner
from llm_d_lens.auth.context import AuthContextMiddleware, require_route_permission
from llm_d_lens.auth.maintenance import run_maintenance
from llm_d_lens.auth.master_key import ensure_master_key
from llm_d_lens.auth.router import router as auth_router
from llm_d_lens.auth.routes import validate_route_registry
from llm_d_lens.auth.service import default_service as default_auth_service
from llm_d_lens.auth.settings import get_settings as get_auth_settings
from llm_d_lens.auth.settings import settings as auth_settings
from llm_d_lens.capacity.router import router as capacity_router
from llm_d_lens.cluster.bootstrap.router import router as cluster_bootstrap_router
from llm_d_lens.cluster.errors import ClusterOverviewError
from llm_d_lens.cluster.router import legacy_router
from llm_d_lens.cluster.router import router as cluster_management_router
from llm_d_lens.configuration import router as configuration_router
from llm_d_lens.configuration.router import legacy_router as legacy_configuration_router
from llm_d_lens.dashboard.collector import reset_collector
from llm_d_lens.dashboard.router import router as dashboard_router
from llm_d_lens.db.system_router import router as system_database_router
from llm_d_lens.deploy.guide_planning import router as guide_planning_router
from llm_d_lens.deploy.legacy_poc import router as legacy_poc_router
from llm_d_lens.deploy.remote_deploy import router as remote_deploy_router
from llm_d_lens.deploy.router import router as deploy_router
from llm_d_lens.evaluation import router as evaluate_router
from llm_d_lens.hardware.router import router as hardware_router
from llm_d_lens.infrastructure.kubernetes import router as cluster_router
from llm_d_lens.infrastructure.network import hostinfo as _hostinfo
from llm_d_lens.mcp.router import router as mcp_router
from llm_d_lens.mcp_v2.router import router as mcp_v2_router
from llm_d_lens.model_cache.router import router as model_cache_router
from llm_d_lens.model_service.router import admin_router as model_service_admin_router
from llm_d_lens.model_service.router import internal_router as model_service_internal_router
from llm_d_lens.model_service.router import public_router as model_service_public_router
from llm_d_lens.model_service.router import router as model_service_router
from llm_d_lens.monitoring.accelerator import router as accelerator_router
from llm_d_lens.monitoring.cluster_stack import router as cluster_monitoring_stack_router
from llm_d_lens.monitoring.deployment import router as deployment_monitoring_router
from llm_d_lens.monitoring.gpu_driver import router as gpu_driver_router
from llm_d_lens.monitoring.profiling import router as profiling_router
from llm_d_lens.playground.chat import router as playground_router
from llm_d_lens.simulation import router as simulation_router
from llm_d_lens.simulation.candidate_search import router as candidate_search_router
from llm_d_lens.storage.router import router as storage_router
from llm_d_lens.versions.router import router as versions_router

app = FastAPI(
    title="Lens API",
    description="Model, inference, evaluation and cluster management backend for Lens",
    version="1.0.0",
    docs_url="/api/simulation/docs" if auth_settings.expose_api_docs else None,
    redoc_url="/api/simulation/redoc" if auth_settings.expose_api_docs else None,
    openapi_url="/api/simulation/openapi.json" if auth_settings.expose_api_docs else None,
    dependencies=[Depends(require_route_permission)],
)
install_problem_handlers(app)
app.add_middleware(AuthContextMiddleware)
app.add_middleware(WebEntryMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.middleware("http")
async def _remember_serving_port(request, call_next):
    """Record the port this backend is actually listening on (from the ASGI
    scope) so Gateway ext_authz can build the exact ``host:port`` a cluster must
    reach Lens on, without any user-configured or env-var port."""
    server = request.scope.get("server")
    if server and len(server) > 1:
        _hostinfo.remember_serving_port(server[1])
    return await call_next(request)


# Seed the serving port from the server's own launch args so startup/background
# Gateway reconciles build the right `host:port` before the first request.
_hostinfo.remember_serving_port(_hostinfo.serving_port_from_argv())


@app.exception_handler(ClusterOverviewError)
async def cluster_overview_problem(_request, error: ClusterOverviewError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=error.problem(),
        media_type="application/problem+json",
    )


_maintenance_task: asyncio.Task[None] | None = None
_maintenance_stop: asyncio.Event | None = None
_model_health_task: asyncio.Task[None] | None = None
_model_health_stop: asyncio.Event | None = None
_model_reconcile_task: asyncio.Task[None] | None = None
_model_reconcile_stop: asyncio.Event | None = None
_gateway_exposure_task: asyncio.Task[None] | None = None
_gateway_exposure_stop: asyncio.Event | None = None
_gateway_usage_task: asyncio.Task[None] | None = None
_gateway_usage_stop: asyncio.Event | None = None
_welcome_check_task: asyncio.Task[None] | None = None
_welcome_check_stop: asyncio.Event | None = None
MODEL_HEALTH_INTERVAL_SECONDS = float(os.environ.get("LENS_MODEL_HEALTH_INTERVAL_SECONDS", "10"))
MODEL_RECONCILE_INTERVAL_SECONDS = float(os.environ.get("LENS_MODEL_RECONCILE_INTERVAL_SECONDS", "300"))
GATEWAY_EXPOSURE_INTERVAL_SECONDS = float(os.environ.get("LENS_GATEWAY_EXPOSURE_INTERVAL_SECONDS", "60"))
GATEWAY_USAGE_INTERVAL_SECONDS = float(os.environ.get("LENS_GATEWAY_USAGE_INTERVAL_SECONDS", "60"))
WELCOME_CHECK_INTERVAL_SECONDS = float(os.environ.get("LENS_WELCOME_CHECK_INTERVAL_SECONDS", "60"))
logger = logging.getLogger(__name__)


async def _run_gateway_exposure(stop_event: asyncio.Event) -> None:
    """Keep each cluster's public 0.0.0.0 Gateway tunnel established.

    Owned by the long-running backend so the tunnels survive (and are rebuilt)
    across restarts; ``ensure_exposures`` reuses existing forwards.
    """
    from llm_d_lens.model_service.gateway_ops import GatewayOpsService  # noqa: PLC0415

    service = GatewayOpsService()
    while not stop_event.is_set():
        try:
            await service.ensure_exposures()
        except Exception:  # noqa: BLE001 - exposure must not kill the loop
            logger.exception("model-service gateway exposure failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=GATEWAY_EXPOSURE_INTERVAL_SECONDS)


async def _run_gateway_usage(stop_event: asyncio.Event) -> None:
    """Scrape each cluster's EPP token counters and write usage deltas.

    llm-d Gateway Mode meters per request via the EPP ``fairness_id`` label, so
    the backend periodically diffs those counters into ``usage_records``.
    """
    from llm_d_lens.model_service.service import default_service  # noqa: PLC0415
    from llm_d_lens.model_service.usage_sync import GatewayUsageSync  # noqa: PLC0415

    sync = GatewayUsageSync(service=default_service())
    while not stop_event.is_set():
        try:
            await sync.sync()
        except Exception:  # noqa: BLE001 - usage sync must not kill the loop
            logger.exception("model-service gateway usage sync failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=GATEWAY_USAGE_INTERVAL_SECONDS)


async def _run_welcome_check(stop_event: asyncio.Event) -> None:
    """Refresh the persisted Welcome issue snapshot for the whole installation."""
    from llm_d_lens.dashboard.service import default_service as default_dashboard_service  # noqa: PLC0415

    service = default_dashboard_service()
    while not stop_event.is_set():
        try:
            await service.run_snapshot_check()
        except Exception:  # noqa: BLE001 - a failed check must not kill the loop
            logger.exception("welcome snapshot check failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=WELCOME_CHECK_INTERVAL_SECONDS)


async def _run_model_health(stop_event: asyncio.Event) -> None:
    """Periodically probe member backends and refresh their health/status."""
    from llm_d_lens.model_service.gateway_ops import GatewayOpsService  # noqa: PLC0415

    service = GatewayOpsService()
    while not stop_event.is_set():
        try:
            await service.probe_members()
        except Exception:  # noqa: BLE001 - a probe failure must not kill the loop
            logger.exception("model-service health probe failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=MODEL_HEALTH_INTERVAL_SECONDS)


async def _run_model_reconcile(stop_event: asyncio.Event) -> None:
    """Bootstrap and self-heal the shared llm-d Gateway and per-model routes.

    Reconciles every cluster that has published members; the underlying applies
    are idempotent, so repeated runs are safe.
    """
    from llm_d_lens.model_service.gateway_ops import GatewayOpsService  # noqa: PLC0415

    service = GatewayOpsService()
    while not stop_event.is_set():
        try:
            cluster_ids = sorted(
                {member.cluster_id for member in service.members.list_all() if member.status != "disabled"}
            )
            for cluster_id in cluster_ids:
                await service.reconcile_cluster(cluster_id)
        except Exception:  # noqa: BLE001 - reconcile must not kill the loop
            logger.exception("model-service reconcile failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=MODEL_RECONCILE_INTERVAL_SECONDS)


@app.on_event("startup")
async def _auth_startup() -> None:
    global _maintenance_task, _maintenance_stop, _model_health_task, _model_health_stop
    global _model_reconcile_task, _model_reconcile_stop
    global _gateway_exposure_task, _gateway_exposure_stop
    global _gateway_usage_task, _gateway_usage_stop
    global _welcome_check_task, _welcome_check_stop
    service = default_auth_service()
    service.ensure_builtin_roles()
    current_settings = get_auth_settings()
    ensure_master_key(current_settings)
    initial = ensure_initial_admin(service, current_settings)
    if initial is not None:
        banner = format_initial_admin_banner(initial)
        logger.warning(banner)
        print(banner, flush=True)
    service.sweep_sessions()
    if current_settings.expose_api_docs:
        problems = validate_route_registry(app)
        if problems:
            raise RuntimeError("auth route permission registry is invalid: " + "; ".join(problems))
    _maintenance_stop = asyncio.Event()
    _maintenance_task = asyncio.create_task(run_maintenance(service, stop_event=_maintenance_stop))
    if MODEL_HEALTH_INTERVAL_SECONDS > 0:
        _model_health_stop = asyncio.Event()
        _model_health_task = asyncio.create_task(_run_model_health(_model_health_stop))
    if MODEL_RECONCILE_INTERVAL_SECONDS > 0:
        _model_reconcile_stop = asyncio.Event()
        _model_reconcile_task = asyncio.create_task(_run_model_reconcile(_model_reconcile_stop))
    if GATEWAY_EXPOSURE_INTERVAL_SECONDS > 0:
        _gateway_exposure_stop = asyncio.Event()
        _gateway_exposure_task = asyncio.create_task(_run_gateway_exposure(_gateway_exposure_stop))
    if GATEWAY_USAGE_INTERVAL_SECONDS > 0:
        _gateway_usage_stop = asyncio.Event()
        _gateway_usage_task = asyncio.create_task(_run_gateway_usage(_gateway_usage_stop))
    if WELCOME_CHECK_INTERVAL_SECONDS > 0:
        _welcome_check_stop = asyncio.Event()
        _welcome_check_task = asyncio.create_task(_run_welcome_check(_welcome_check_stop))


@app.on_event("shutdown")
async def _auth_shutdown() -> None:
    global _maintenance_task, _maintenance_stop, _model_health_task, _model_health_stop
    global _model_reconcile_task, _model_reconcile_stop
    global _gateway_exposure_task, _gateway_exposure_stop
    global _gateway_usage_task, _gateway_usage_stop
    global _welcome_check_task, _welcome_check_stop
    if _maintenance_stop is not None:
        _maintenance_stop.set()
    if _maintenance_task is not None:
        # Shutdown must not fail on maintenance errors.
        with contextlib.suppress(Exception):
            await _maintenance_task
        _maintenance_task = None
    _maintenance_stop = None
    if _model_health_stop is not None:
        _model_health_stop.set()
    if _model_health_task is not None:
        try:
            await _model_health_task
        except Exception:  # noqa: BLE001 - shutdown must not fail on health probe
            logger.warning("model-service health task did not stop cleanly", exc_info=True)
        _model_health_task = None
    _model_health_stop = None
    if _model_reconcile_stop is not None:
        _model_reconcile_stop.set()
    if _model_reconcile_task is not None:
        try:
            await _model_reconcile_task
        except Exception:  # noqa: BLE001 - shutdown must not fail on reconcile
            logger.warning("model-service reconcile task did not stop cleanly", exc_info=True)
        _model_reconcile_task = None
    _model_reconcile_stop = None
    if _gateway_exposure_stop is not None:
        _gateway_exposure_stop.set()
    if _gateway_exposure_task is not None:
        try:
            await _gateway_exposure_task
        except Exception:  # noqa: BLE001 - shutdown must not fail on exposure
            logger.warning("gateway exposure task did not stop cleanly", exc_info=True)
        _gateway_exposure_task = None
    _gateway_exposure_stop = None
    if _gateway_usage_stop is not None:
        _gateway_usage_stop.set()
    if _gateway_usage_task is not None:
        try:
            await _gateway_usage_task
        except Exception:  # noqa: BLE001 - shutdown must not fail on usage sync
            logger.warning("gateway usage task did not stop cleanly", exc_info=True)
        _gateway_usage_task = None
    _gateway_usage_stop = None
    if _welcome_check_stop is not None:
        _welcome_check_stop.set()
    if _welcome_check_task is not None:
        try:
            await _welcome_check_task
        except Exception:  # noqa: BLE001 - shutdown must not fail on snapshot check
            logger.warning("welcome snapshot task did not stop cleanly", exc_info=True)
        _welcome_check_task = None
    _welcome_check_stop = None
    reset_collector()


app.include_router(simulation_router)
app.include_router(candidate_search_router)
app.include_router(agentic_deployment_router)
app.include_router(capacity_router)
app.include_router(auth_router)
app.include_router(aic_router)
app.include_router(cluster_router)
app.include_router(system_database_router)
app.include_router(cluster_management_router)
app.include_router(cluster_bootstrap_router)
app.include_router(legacy_router)
app.include_router(configuration_router)
app.include_router(legacy_configuration_router)
app.include_router(deploy_router)
app.include_router(legacy_poc_router)
app.include_router(guide_planning_router)
app.include_router(mcp_router)
app.include_router(mcp_v2_router)
app.include_router(playground_router)
app.include_router(web_router)
app.include_router(remote_deploy_router)
app.include_router(dashboard_router)
app.include_router(speech_models_router)
app.include_router(evaluate_router)
app.include_router(hardware_router)
app.include_router(cluster_monitoring_stack_router)
app.include_router(deployment_monitoring_router)
app.include_router(profiling_router)
app.include_router(accelerator_router)
app.include_router(gpu_driver_router)
app.include_router(storage_router)
app.include_router(model_cache_router)
app.include_router(ai_providers_router)
app.include_router(model_service_router)
app.include_router(model_service_admin_router)
app.include_router(model_service_internal_router)
app.include_router(model_service_public_router)
app.include_router(versions_router)


@app.get("/api/health", tags=["system"])
@app.get("/api/simulation/health", include_in_schema=False)
@app.get("/healthz", include_in_schema=False)
async def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
    }
