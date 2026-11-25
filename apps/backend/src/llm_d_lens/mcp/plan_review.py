"""Reviewed deployment plans: what Lens freezes for a DisaggregatedSet deployment, and how a rejection reads."""

from __future__ import annotations

from typing import Any

from .errors import ToolError
from .lens_api import LensApiError
from .registry import upstream_error

MAX_FINDINGS = 10
DEPLOYMENT_REVIEW = (
    "This is a prefill/decode DisaggregatedSet, which Lens only deploys from a reviewed plan. Show the user the "
    "preflight result, its warnings and unknown constraints, and the namespace, and ask them to approve before "
    "you create it, unless they already told you to go ahead."
)
EXPERIMENT_REVIEW = (
    "This experiment deploys a prefill/decode DisaggregatedSet, so Lens holds it until its plans are confirmed. "
    "Show the user the preflight results and ask them to review the plans (the Lens Web UI shows them under "
    "Evaluation > the experiment > Review deployment plans). Call experiment_confirm once the user has reviewed "
    "them or told you to go ahead."
)


def _messages(items: Any) -> list[str]:
    return [str(item.get("message") if isinstance(item, dict) else item)[:300] for item in (items or [])][:MAX_FINDINGS]


def review_view(plan: dict[str, Any]) -> dict[str, Any]:
    """The part of a Lens plan an agent needs to show the user; manifests stay in Lens."""
    preflight = plan.get("preflight") or {}
    return {
        "plan_id": plan["id"],
        "plan_digest": plan["digest"],
        "preflight": {
            "status": preflight.get("status"),
            "errors": _messages(preflight.get("errors")),
            "warnings": _messages(preflight.get("warnings")),
            "unknown_constraints": _messages(preflight.get("unmodeled")),
        },
        "workloads": [
            {"namespace": item.get("namespace"), "manifest_checksum": item.get("manifestChecksum")}
            for item in plan.get("cases") or []
        ],
    }


def plan_rejection(error: LensApiError) -> ToolError:
    """Lens refused to execute a reviewed plan: changed, unavailable, failed or not acknowledged."""
    if error.status not in (409, 422):
        return upstream_error(error)
    message = error.message()
    lowered = message.lower()
    if "acknowledged" in lowered:
        return ToolError(
            "unknown_not_acknowledged",
            message,
            hint="Preflight could not check some constraints. Show them to the user and, if they accept the risk, "
            "repeat the call with acknowledge_unknown=true.",
        )
    if "digest" in lowered or "prepare a new plan" in lowered or "unavailable" in lowered or "not available" in lowered:
        return ToolError(
            "plan_stale",
            message,
            hint="The plan changed or expired. Call deployment_plan again, review it, and use its new plan_id and "
            "plan_digest.",
        )
    return ToolError("plan_rejected", message, hint="Lens rejected the reviewed plan; the message says why.")
