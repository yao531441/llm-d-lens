"""Per-call context handed to every tool handler."""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import Ledger
from .lens_api import LensApi


@dataclass(frozen=True)
class ToolContext:
    api: LensApi
    ledger: Ledger
    owner: str
