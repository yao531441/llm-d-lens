"""Lens-owned verification tasks persisted in execution diagnostics."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from llm_d_lens.deploy.application import deployment_run_manager
from llm_d_lens.deploy.contracts import utcnow
from llm_d_lens.deploy.executions import ensure_execution_endpoint, get_execution_context
from llm_d_lens.deploy.usage import register_usage_probe
from llm_d_lens.deploy.verification_checks import VerificationRequest, run_checks

_tasks: dict[str, asyncio.Task] = {}
_store = deployment_run_manager.store
ACTIVE = {"queued", "running"}
TIMEOUT_SECONDS = 120


def list_verifications(execution_id: str) -> list[dict]:
    execution = _store.get_execution(execution_id)
    if execution is None:
        raise ValueError("deployment execution not found")
    return list((execution.diagnostics.value if execution.diagnostics else {}).get("verifications", []))


def get_verification(execution_id: str, verification_id: str) -> dict:
    for record in list_verifications(execution_id):
        if record["id"] == verification_id:
            return record
    raise ValueError("deployment verification not found")


def start_verification(execution_id: str, request: VerificationRequest) -> dict:
    context = get_execution_context(execution_id)
    if context.status != "ready" or not context.model:
        raise ValueError("deployment must be ready and have a model before verification")
    record = {
        "id": str(uuid4()),
        "execution_id": execution_id,
        "status": "queued",
        "request": request.model_dump(),
        "created_at": utcnow().isoformat(),
        "requests_sent": 0,
        "requests_completed": 0,
        "requests_succeeded": 0,
        "output_tokens": 0,
        "max_output_tokens": request.requests * 16,
    }
    _store.update_execution_verification(execution_id, record, create=True)
    task = asyncio.create_task(_execute(execution_id, record["id"], request, context.model))
    _tasks[record["id"]] = task
    task.add_done_callback(lambda _: _tasks.pop(record["id"], None))
    return record


async def _execute(execution_id: str, identifier: str, request: VerificationRequest, model: str) -> None:
    def update(**changes):
        _store.update_execution_verification(execution_id, {"id": identifier, **changes})

    update(status="running", started_at=utcnow().isoformat())
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            endpoint = await ensure_execution_endpoint(execution_id)
            result = await run_checks(endpoint.endpoint, model, request, update)
        update(status="succeeded", finished_at=utcnow().isoformat(), **result)
    except TimeoutError:
        update(
            status="failed",
            finished_at=utcnow().isoformat(),
            error=f"Verification did not finish within {TIMEOUT_SECONDS} seconds; the counters show how far it got.",
        )
    except asyncio.CancelledError:
        update(
            status="failed",
            finished_at=utcnow().isoformat(),
            error="Verification was cancelled because the backend is shutting down.",
        )
        raise
    except Exception:
        update(
            status="failed",
            finished_at=utcnow().isoformat(),
            error="Verification could not complete; inspect deployment readiness and connectivity.",
        )


def fail_interrupted_verifications() -> None:
    """Fail tasks left active by a previous backend process; requests are never replayed."""
    for execution in _store.list_executions():
        for record in (execution.diagnostics.value if execution.diagnostics else {}).get("verifications", []):
            if record["status"] in ACTIVE:
                _store.update_execution_verification(
                    execution.execution_id,
                    {
                        "id": record["id"],
                        "status": "failed",
                        "finished_at": utcnow().isoformat(),
                        "error": "Verification interrupted by a backend restart; requests were not replayed.",
                    },
                )


def active_usage(execution_id: str) -> str | None:
    if _store.get_execution(execution_id) is None:
        return None
    if any(record["status"] in ACTIVE for record in list_verifications(execution_id)):
        return "deployment verification is active; wait for its result before teardown"
    return None


register_usage_probe("verification", active_usage)
