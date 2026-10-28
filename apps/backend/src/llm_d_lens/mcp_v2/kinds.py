"""Per-kind behaviour behind ``list_resources`` and ``delete_resources``.

Each domain registers one ``KindHandler`` for the ledger kind it creates, so the two generic tools never need to
know how a deployment, an experiment or a model service is actually removed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .context import ToolContext
from .ledger import KIND_PREFIXES, Resource

# Removes the remote objects (idempotent: a remote 404 counts as success). Raise ``ToolError("in_use", ...)``
# when something still depends on the resource and ``cascade`` is false.
DeleteFn = Callable[[ToolContext, Resource, bool], Awaitable[None]]
# Live summary merged into the list_resources entry (status, endpoint, ...).
DescribeFn = Callable[[ToolContext, Resource], Awaitable[dict[str, Any]]]
# Things that block deletion or would be removed with it: ``[{"id", "kind", "name", "reason"}]``.
DependentsFn = Callable[[ToolContext, Resource], Awaitable[list[dict[str, Any]]]]
# Visible resources of this kind that are not owned by the caller (read-only); entries carry ``ownership``.
ListOthersFn = Callable[[ToolContext, str | None], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class KindHandler:
    kind: str
    delete: DeleteFn
    describe: DescribeFn | None = None
    dependents: DependentsFn | None = None
    list_others: ListOthersFn | None = None


_HANDLERS: dict[str, KindHandler] = {}


def register_kind(
    kind: str,
    *,
    delete: DeleteFn,
    describe: DescribeFn | None = None,
    dependents: DependentsFn | None = None,
    list_others: ListOthersFn | None = None,
) -> None:
    if kind not in KIND_PREFIXES:
        raise ValueError(f"unknown resource kind {kind}")
    _HANDLERS[kind] = KindHandler(kind, delete, describe, dependents, list_others)


def handler_for(kind: str) -> KindHandler | None:
    from . import tools  # noqa: F401  (importing registers every kind)

    return _HANDLERS.get(kind)


def all_kinds() -> list[str]:
    from . import tools  # noqa: F401

    return sorted(_HANDLERS)
