"""Guard for FakeApi-based tests: every call a tool makes must exist, with that HTTP method, in the real backend."""

from __future__ import annotations

from starlette.routing import Match

from llm_d_lens.api.main import app
from llm_d_lens.mcp_v2.lens_api import _UNVERSIONED

# Prefixes the backend serves without the /api/v1 prefix that HttpLensApi/InProcessLensApi add.
_ALIASED = {"simulation"}


def backend_path(path: str) -> str:
    root = path.lstrip("/").split("/", 1)[0]
    return f"/api{path}" if root in _UNVERSIONED | _ALIASED else f"/api/v1{path}"


def route_problem(method: str, path: str) -> str | None:
    full = backend_path(path)
    method_mismatch = False
    for route in app.router.routes:
        match, _ = route.matches({"type": "http", "path": full, "method": method})
        if match == Match.FULL:
            return None
        method_mismatch = method_mismatch or match == Match.PARTIAL
    return f"{method} {full} {'exists with another HTTP method' if method_mismatch else 'does not exist'}"


def record_route_problems(api) -> list[str]:
    """Wrap ``api.request`` and collect calls whose method or path the real backend does not serve.

    Collected rather than raised so a tool's own error handling cannot swallow the finding.
    """
    problems: list[str] = []
    original = api.request

    async def checked(method, path, **kwargs):
        problem = route_problem(method, path)
        if problem:
            problems.append(problem)
        return await original(method, path, **kwargs)

    api.request = checked
    return problems
