"""Cross-module usage guards for Deploy-owned deployments.

Stopping or deleting a deployment must not break work that is actively using it. Consumer
modules (Evaluate today, others later) register a probe here instead of Deploy
importing them, which keeps the dependency direction one-way: consumers depend
on Deploy, never the reverse.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

UsageProbe = Callable[[str], str | None]

_probes: dict[str, UsageProbe] = {}
_exempt: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar("usage_exempt", default=frozenset())


class DeploymentInUseError(ValueError):
    """A consumer still needs the deployment or its usage cannot be verified."""


def require_unused(execution_id: str) -> None:
    """Reject destructive lifecycle operations while a registered consumer uses the deployment."""
    if reason := active_usage_reason(execution_id):
        raise DeploymentInUseError(reason)


@contextlib.contextmanager
def exempt_usage(*names: str) -> Iterator[None]:
    """Skip the named probes for the current task.

    A consumer that cleans up its own deployment (Evaluate, when a case ends or fails) is itself
    what its probe reports, so it must not be blocked by that probe. It still answers to every
    other consumer.
    """
    token = _exempt.set(_exempt.get() | frozenset(names))
    try:
        yield
    finally:
        _exempt.reset(token)


def register_usage_probe(name: str, probe: UsageProbe) -> None:
    """Register a probe returning a human-readable reason when a deployment is busy."""
    _probes[name] = probe


def active_usage_reason(execution_id: str) -> str | None:
    """Return the first reason a deployment is still in use, if any.

    A failing probe never silently unblocks a lifecycle operation: it is reported as a
    conflict so a deployment in unknown use is not destroyed.
    """
    for name, probe in _probes.items():
        if name in _exempt.get():
            continue
        try:
            reason = probe(execution_id)
        except Exception:
            logger.exception("Deployment usage probe %s failed", name)
            return f"usage of this deployment could not be verified by {name}"
        if reason:
            return reason
    return None
