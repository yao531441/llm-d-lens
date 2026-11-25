"""Structured tool errors returned to the agent as ``isError`` results."""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """A failure the agent can act on: stable ``code``, offending ``field``, and a next-step ``hint``."""

    def __init__(self, code: str, message: str, *, field: str | None = None, hint: str | None = None, **extra: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.hint = hint
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field:
            body["field"] = self.field
        if self.hint:
            body["hint"] = self.hint
        body.update(self.extra)
        return {"error": body}
