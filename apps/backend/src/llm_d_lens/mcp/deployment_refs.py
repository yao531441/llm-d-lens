"""Resolve a deployment id (a Lens deployment run id, or an execution id) to its run, case and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import ToolContext
from .errors import ToolError
from .lens_api import LensApiError
from .path_ids import check_id


@dataclass(frozen=True)
class DeploymentRef:
    execution_id: str | None
    run_id: str | None
    execution: dict[str, Any] = field(default_factory=dict)
    case_id: str | None = None

    @property
    def owned(self) -> bool:
        return self.execution.get("ownership") == "self"

    @property
    def public_id(self) -> str:
        return self.run_id or self.execution_id or ""


def not_found(deployment_id: str) -> ToolError:
    return ToolError(
        "deployment_not_found",
        f"No deployment '{deployment_id}' is visible to you.",
        field="deployment_id",
        hint="Call resource_list(kinds=['deployment'], owner='all') to see valid ids.",
    )


def require_owned(ref: DeploymentRef, deployment_id: str, action: str) -> None:
    """The agent-only ownership rule: Lens must allow the action and the caller must own the deployment.

    Lens RBAC alone would let any maintainer act on every deployment in a cluster; this restriction exists for AI
    agents and is not a Lens rule, so the Web UI does not share it.
    """
    capability = {"stopped": "stop", "deleted": "delete", "published": None, "tested": None}[action]
    ownership = ref.execution.get("ownership")
    if ownership == "self" and (capability is None or (ref.execution.get("capabilities") or {}).get(capability)):
        return
    raise ToolError(
        "not_owner" if ownership == "other" else "owner_unknown",
        f"Deployment '{deployment_id}' cannot be {action} by you: "
        + ("it belongs to another user." if ownership == "other" else "Lens did not confirm you own it."),
        field="deployment_id",
        hint="Only deployments you created and may manage through Lens can be changed through this server.",
    )


def latest_case(run: dict[str, Any]) -> dict[str, Any] | None:
    """The newest attempt of the run's single case chain (restarts append attempts)."""
    cases = run.get("cases") or []
    chains = {(case.get("ordinal", 0), case.get("source_configuration_ordinal", 0)) for case in cases}
    if len(chains) > 1:
        raise ToolError(
            "conflict_state",
            "This run contains multiple deployment cases; use the exact Lens execution id.",
            field="deployment_id",
        )
    return max(cases, key=lambda case: case.get("attempt") or 0) if cases else None


async def _get_or_none(ctx: ToolContext, path: str) -> dict[str, Any] | None:
    try:
        return await ctx.api.request("GET", path)
    except LensApiError as error:
        if error.status == 404:
            return None
        raise


async def get_run(ctx: ToolContext, run_id: str) -> dict[str, Any] | None:
    return await _get_or_none(ctx, f"/deployments/runs/{run_id}")


async def resolve(ctx: ToolContext, deployment_id: str) -> DeploymentRef:
    """A run id resolves through the run's newest case; anything else is read as an execution id."""
    check_id(deployment_id, "deployment_id")
    run = await get_run(ctx, deployment_id)
    if run is not None:
        case = latest_case(run) or {}
        execution_id = case.get("execution_id")
        execution = await _get_or_none(ctx, f"/deployments/executions/{execution_id}") if execution_id else None
        return DeploymentRef(execution_id, run["id"], execution or {}, case.get("id"))
    execution = await _get_or_none(ctx, f"/deployments/executions/{deployment_id}")
    if execution is None:
        raise not_found(deployment_id)
    return DeploymentRef(deployment_id, execution.get("run_id"), execution, execution.get("case_id"))
