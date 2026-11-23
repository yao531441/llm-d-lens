"""Per-kind behaviour behind ``resource_list`` and ``resource_delete``.

Each domain registers one ``KindHandler``, so the two generic tools never need to know how a deployment, an
experiment or a model service is actually listed or removed. Resources are identified by their Lens ids; nothing is
recorded locally.

An entry is ``{id, kind, name, status, ownership, cluster_id, created_at, ...}`` where ``ownership`` is

* ``owned``: Lens records an owner and it is the caller,
* ``shared``: Lens records another owner,
* ``unknown``: Lens records an owner for this kind but not for this resource,
* ``unrecorded``: Lens records no owner for this kind (storage volumes, model caches, simulations).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .context import ToolContext

KINDS = ("deployment", "experiment", "model_cache", "storage_volume", "model_service", "simulation")

ListFn = Callable[[ToolContext, str | None], Awaitable[list[dict[str, Any]]]]
# The entry for one id, or None when Lens has no such resource.
GetFn = Callable[[ToolContext, str], Awaitable[dict[str, Any] | None]]
# Removes the remote objects and returns what went with it (``[{"kind", "id"}]``). A remote 404 counts as success.
# Raise ``ToolError`` when something still uses the resource or the caller may not delete it.
DeleteFn = Callable[[ToolContext, str], Awaitable[list[dict[str, str]]]]


@dataclass(frozen=True)
class KindHandler:
    kind: str
    list: ListFn
    get: GetFn
    delete: DeleteFn


_HANDLERS: dict[str, KindHandler] = {}


def register_kind(kind: str, *, list: ListFn, get: GetFn, delete: DeleteFn) -> None:  # noqa: A002
    if kind not in KINDS:
        raise ValueError(f"unknown resource kind {kind}")
    _HANDLERS[kind] = KindHandler(kind, list, get, delete)


def handler_for(kind: str) -> KindHandler:
    from . import tools  # noqa: F401  (importing registers every kind)

    return _HANDLERS[kind]


def owner_state(ctx: ToolContext, owner_user_id: str | None) -> str:
    """``ownership`` of a resource whose Lens record carries ``owner_user_id``."""
    if not owner_user_id:
        return "owned" if ctx.owner == "system" else "unknown"  # Lens without authentication records no owners
    return "owned" if owner_user_id == ctx.owner else "shared"
