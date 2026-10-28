"""Secret detection for tool arguments and redaction for tool output."""

from __future__ import annotations

import re
from typing import Any

_SECRET = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def find_secret(value: Any, path: str = "arguments") -> str | None:
    """Return a ``path looks like ...`` description of the first secret-shaped string, or None."""
    if isinstance(value, str):
        return f"{path}" if _SECRET.search(value) else None
    if isinstance(value, dict):
        for key, item in value.items():
            found = find_secret(item, f"{key}" if path == "arguments" else f"{path}.{key}")
            if found:
                return found
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = find_secret(item, f"{path}[{index}]")
            if found:
                return found
    return None


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET.sub("***", value)
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
