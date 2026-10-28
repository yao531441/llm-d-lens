"""Orphan namespace cleanup: delete only what Lens itself lists as orphaned and the caller can name with evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .admin_guard import refuse_protected
from .context import ToolContext
from .errors import ToolError

MAX_LISTED = 100


def _instant(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def list_orphans(ctx: ToolContext, cluster_id: str) -> dict[str, Any]:
    page = await ctx.api.request("GET", "/deployments/orphans", params={"cluster_id": cluster_id})
    items = page.get("items") or []
    return {
        "total": len(items),
        "candidates": [
            {"name": item["name"], "created_at": item.get("createdAt"), "phase": item.get("phase")}
            for item in items[:MAX_LISTED]
        ],
    }


async def delete_orphans(ctx: ToolContext, cluster_id: str, namespaces: list[tuple[str, str]]) -> dict[str, Any]:
    """Delete namespaces named with their creation time; every pair must match Lens's current orphan listing."""
    for name, _ in namespaces:
        refuse_protected(name, field="namespaces")
    page = await ctx.api.request("GET", "/deployments/orphans", params={"cluster_id": cluster_id})
    listed = {item["name"]: _instant(item.get("createdAt")) for item in page.get("items") or []}
    for name, created_at in namespaces:
        if name not in listed:
            raise ToolError(
                "conflict_state",
                f"Lens does not list '{name}' as an orphan on cluster {cluster_id}; nothing was deleted.",
                field="namespaces",
                hint="Call plan_cluster_component(component='orphan-namespaces') for the current candidates.",
            )
        if _instant(created_at) != listed[name]:
            raise ToolError(
                "conflict_state",
                f"'{name}' was created at a different time than given, so it may be a newer namespace reusing the "
                "name; nothing was deleted.",
                field="namespaces",
                hint="Re-read the candidates and pass the exact created_at.",
            )
    outcome = await ctx.api.request(
        "POST",
        "/deployments/orphans/clean",
        body={"clusterId": cluster_id, "namespaces": [name for name, _ in namespaces]},
        timeout=300,
    )
    failed = outcome.get("failed") or []
    return {
        "cluster_id": cluster_id,
        "cleaned": outcome.get("cleaned") or [],
        "failed": [item.get("name") for item in failed],
        "untrusted_text": [f"{item.get('name')}: {item.get('error')}" for item in failed],
    }
