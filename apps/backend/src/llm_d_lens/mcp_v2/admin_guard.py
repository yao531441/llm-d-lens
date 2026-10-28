"""Safety rules shared by the cluster-admin tools: protected namespaces and identifier shape."""

from __future__ import annotations

from .errors import ToolError

CLUSTER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# Infrastructure the cluster-admin tools must never act on by name, whatever the caller passes.
PROTECTED_NAMESPACES = frozenset(
    {
        "default",
        "kube-system",
        "kube-public",
        "kube-node-lease",
        "monitoring",
        "llm-d-monitoring",
        "intel-xpumd",
        "lws-system",
        "istio-system",
    }
)
PROTECTED_PREFIXES = ("kube-", "lens-gateway")


def is_protected_namespace(name: str, extra: frozenset[str] = frozenset()) -> bool:
    return name in PROTECTED_NAMESPACES or name in extra or name.startswith(PROTECTED_PREFIXES)


def refuse_protected(name: str, *, field: str, extra: frozenset[str] = frozenset()) -> None:
    """Raise before any Lens call when ``name`` is shared infrastructure."""
    if is_protected_namespace(name, extra):
        raise ToolError(
            "protected_resource",
            f"Namespace '{name}' is shared infrastructure and is not touched by agent tools.",
            field=field,
            hint="Ask the cluster administrator to handle it outside the agent interface.",
        )
