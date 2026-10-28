"""Per-call context handed to every tool handler."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .ledger import Ledger
from .lens_api import LensApi


@dataclass(frozen=True)
class ToolContext:
    api: LensApi
    ledger: Ledger
    owner: str
    scopes: frozenset[str] = frozenset({"core"})


def scopes_from_env() -> frozenset[str]:
    """Profiles this server exposes; ``LENS_MCP_V2_PROFILES=core,serving`` (default: core)."""
    raw = os.environ.get("LENS_MCP_V2_PROFILES", "core")
    return frozenset(item.strip() for item in raw.split(",") if item.strip()) | {"core"}
