"""Pure shaping of Lens model-service payloads into the agent-facing view."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def is_routable(member: dict[str, Any]) -> bool:
    return member.get("status") == "active" and int(member.get("weight") or 0) > 0


def member_health(member: dict[str, Any]) -> str:
    if member.get("status") == "unhealthy":
        return "unhealthy"
    health = member.get("healthJson")
    if not health:
        return "unknown"
    return "healthy" if health.get("ready") else "unhealthy"


def service_version(group: dict[str, Any], members: list[dict[str, Any]]) -> str:
    """Optimistic-concurrency token: changes whenever the group record, membership, weights or states change."""
    state = [group.get("updatedAt"), group.get("status")]
    state += sorted((m["executionId"], m.get("weight"), m.get("status")) for m in members)
    return "v-" + hashlib.sha256(json.dumps(state, default=str).encode()).hexdigest()[:10]


def service_status(group: dict[str, Any], members: list[dict[str, Any]]) -> str:
    if group.get("status") == "disabled":
        return "disabled"
    routable = [m for m in members if is_routable(m)]
    if not routable:
        return "degraded"
    if all(member_health(m) == "unknown" for m in routable):
        return "publishing"
    if any(member_health(m) == "unhealthy" for m in routable):
        return "degraded"
    return "active"


def member_table(members: list[dict[str, Any]], deployment_ids: dict[str, str]) -> list[dict[str, Any]]:
    """One row per member; ``deployment_ids`` maps Lens execution ids to the caller's ledger ids."""
    total = sum(int(m.get("weight") or 0) for m in members if m.get("status") == "active")
    rows = []
    for member in sorted(members, key=lambda m: m["executionId"]):
        weight = int(member.get("weight") or 0)
        rows.append(
            {
                "deployment_id": deployment_ids.get(member["executionId"], member["executionId"]),
                "owned": member["executionId"] in deployment_ids,
                "weight": weight,
                "status": member.get("status"),
                "health": member_health(member),
                "traffic_share_percent": round(100 * weight / total, 1)
                if total and member.get("status") == "active"
                else 0.0,
            }
        )
    return rows


def service_view(
    group: dict[str, Any],
    members: list[dict[str, Any]],
    *,
    service_id: str | None,
    deployment_ids: dict[str, str],
    base_url: str | None,
    owned: bool,
) -> dict[str, Any]:
    return {
        "service_id": service_id or group["id"],
        "name": group["name"],
        "is_mine": owned,
        "cluster_id": group.get("clusterId"),
        "status": service_status(group, members),
        "enabled": group.get("status") == "active",
        "description": group.get("description") or "",
        "selection_policy": group.get("selectionPolicy"),
        "endpoint": {
            "base_url": base_url,
            "model": group["name"],
            "auth": "bearer service token (created in the Lens UI)",
        }
        if base_url
        else None,
        "members": member_table(members, deployment_ids),
        "version": service_version(group, members),
    }


def connection_examples(base_url: str, model: str) -> dict[str, str]:
    """Client snippets; the key is always a placeholder, never a real token."""
    return {
        "curl": (
            f"curl {base_url}/chat/completions -H 'Authorization: Bearer $LENS_SERVICE_TOKEN' "
            f"-H 'Content-Type: application/json' "
            f'-d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "Hello"}}]}}\''
        ),
        "openai_python": f'OpenAI(base_url="{base_url}", api_key=os.environ["LENS_SERVICE_TOKEN"])  # model="{model}"',
    }
