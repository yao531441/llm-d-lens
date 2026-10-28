import json
from pathlib import Path
from typing import Any

import pytest

from llm_d_lens.mcp_v2.context import ToolContext
from llm_d_lens.mcp_v2.ledger import Ledger
from llm_d_lens.mcp_v2.lens_api import LensApiError

FIXTURES = Path(__file__).parent / "fixtures"


def load_plan() -> dict[str, Any]:
    return json.loads((FIXTURES / "plan_optimized_baseline_xpu.json").read_text())


class FakeApi:
    """Routes ``(METHOD, logical path)`` to canned responses and records every call."""

    def __init__(self, owner: str = "user-a"):
        self.owner = owner
        self.calls: list[tuple[str, str, Any]] = []
        self.routes: dict[tuple[str, str], Any] = {}

    def on(self, method: str, path: str, response: Any) -> None:
        self.routes[(method, path)] = response

    async def principal_id(self) -> str:
        return self.owner

    async def request(self, method, path, *, params=None, body=None, timeout=60):
        self.calls.append((method, path, body if body is not None else params))
        response = self.routes.get((method, path))
        if response is None:
            raise LensApiError(404, {"detail": "not found"}, method, path)
        if isinstance(response, Exception):
            raise response
        return response(params, body) if callable(response) else response

    def called(self, method: str, path: str) -> list[Any]:
        return [payload for m, p, payload in self.calls if (m, p) == (method, path)]


def cluster_routes(api: FakeApi) -> None:
    api.on(
        "GET",
        "/cluster/overview",
        {
            "kubernetes": {
                "hardware": {
                    "accelerators": [{"id": "intel-xpu", "gpuCount": 32, "models": ["B60"]}],
                    "usableGpuCount": 26,
                    "allocatedGpuCount": 6,
                },
                "nodes": [{"name": "n1", "role": "worker", "ready": True, "gpuByProfile": {"intel-xpu": 8}}],
                "components": {"monitoring": {"status": "degraded"}},
            }
        },
    )
    api.on("GET", "/cluster/session", {"sessionId": "s" * 36, "cluster": {"id": "c1", "name": "sprocean-min"}})
    api.on(
        "GET",
        "/guide-planning/catalog",
        {
            "ref": "v1.0.0",
            "guides": [
                {
                    "id": "optimized-baseline",
                    "label": "Optimized Baseline",
                    "accelerators": [{"id": "xpu", "modelServers": [{"id": "vllm", "variants": ["base"]}]}],
                }
            ],
        },
    )


def plan_routes(api: FakeApi) -> None:
    cluster_routes(api)
    api.on("POST", "/guide-planning/plan", load_plan())
    api.on(
        "POST",
        "/configurations/render",
        lambda params, body: {
            "deployable_configuration": {"type": "baseline", "content": {}, "echo": body["render"]["deploymentName"]}
        },
    )


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def ctx(api: FakeApi) -> ToolContext:
    return ToolContext(api=api, ledger=Ledger(":memory:"), owner=api.owner)
