"""Path allow-list for hostPath volumes: Lens reports no allowed prefixes, so the operator configures them."""

from __future__ import annotations

import os
import posixpath

from .errors import ToolError

PREFIXES_ENV = "LENS_MCP_V2_STORAGE_PATH_PREFIXES"


def allowed_prefixes() -> list[str]:
    """Comma-separated absolute directories from the environment; empty (default) allows no hostPath at all."""
    prefixes = []
    for raw in os.environ.get(PREFIXES_ENV, "").split(","):
        value = posixpath.normpath(raw.strip()) if raw.strip().startswith("/") else ""
        if value and value != "/":
            prefixes.append(value)
    return prefixes


def check_host_path(path: str) -> str:
    """Return the normalized path, or raise ``invalid_argument`` naming the allowed prefixes."""
    prefixes = allowed_prefixes()
    if not prefixes:
        raise ToolError(
            "invalid_argument",
            "No hostPath prefixes are allowed on this server, so local-disk volumes cannot be created here.",
            field="path",
            hint=f"An operator must set {PREFIXES_ENV} (comma-separated directories such as /mnt/models), "
            "or use kind='nfs'.",
        )
    if ".." in path.split("/"):
        raise ToolError("invalid_argument", "path must not contain '..'.", field="path", hint=_hint(prefixes))
    normalized = posixpath.normpath(path)
    if not any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes):
        raise ToolError(
            "invalid_argument",
            f"path '{path}' is outside the allowed hostPath prefixes.",
            field="path",
            hint=_hint(prefixes),
        )
    return normalized


def _hint(prefixes: list[str]) -> str:
    return "Allowed prefixes: " + ", ".join(prefixes) + "."
