"""When an unconfirmed ledger row may claim a Lens object found by name (serving and storage reconcile)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from .context import ToolContext
from .ledger import Resource

CLOCK_SKEW_SECONDS = 60
CLAIM_WINDOW_SECONDS = 600  # an unconfirmed create is claimable, and its row deletable-as-pending, only this long


def instant(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()


def row_age(row: Resource) -> float:
    return time.time() - row.created_at


def claim_expired(row: Resource) -> bool:
    return row_age(row) > CLAIM_WINDOW_SECONDS


def may_claim(ctx: ToolContext, row: Resource, lens_object: dict[str, Any]) -> bool:
    """True only for an object Lens created while this row was pending and that no other ledger row owns."""
    created = instant(lens_object.get("createdAt"))
    if created is None:
        return False
    if not row.created_at - CLOCK_SKEW_SECONDS <= created <= row.created_at + CLAIM_WINDOW_SECONDS:
        return False
    return ctx.ledger.find_by_execution(lens_object.get("id") or "") is None
