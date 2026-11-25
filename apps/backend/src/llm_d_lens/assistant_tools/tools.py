"""MCP adapters invoke the same authenticated API used by browser clients."""

from __future__ import annotations

import asyncio
import math
import time
from urllib.parse import urlencode, urlsplit

import httpx
from jsonschema import Draft202012Validator, ValidationError

from llm_d_lens.auth.security import sign_internal
from llm_d_lens.auth.settings import get_settings

from .catalog import catalog, render_path

SPECIAL_ROUTES = {
    "resolve_configurations": ("POST", "/api/configurations/resolve"),
    "render_configuration": ("POST", "/api/configurations/render"),
    "save_configuration": ("POST", "/api/configurations/save"),
    "get_remote_deploy_defaults": ("GET", "/api/remote-deploy/defaults"),
    "get_remote_deploy_fingerprint": ("POST", "/api/remote-deploy/fingerprint"),
    "test_remote_deploy_target": ("POST", "/api/remote-deploy/test"),
    "start_remote_deploy": ("POST", "/api/remote-deploy/start"),
    "get_remote_deploy_status": ("POST", "/api/remote-deploy/status"),
    "get_remote_deploy_logs": ("POST", "/api/remote-deploy/logs"),
    "teardown_remote_deploy": ("POST", "/api/remote-deploy/teardown"),
    "prepare_guide_plan": ("POST", "/api/guide-planning/prepare"),
    "plan_guide_deployment": ("POST", "/api/guide-planning/plan"),
    "check_candidate_support": ("POST", "/api/candidate-support"),
    "search_candidates": ("POST", "/api/candidate-search"),
    "estimate_vllm_capacity": ("POST", "/api/v1/capacity/estimate"),
    "get_guide_catalog": ("GET", "/api/guide-planning/catalog"),
}
STATUS_KEYS = ("status", "phase", "state", "ready", "condition")
TERMINAL = (
    "ready",
    "succeeded",
    "success",
    "completed",
    "complete",
    "done",
    "finished",
    "failed",
    "failure",
    "error",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
    "aborted",
)
PENDING = ("pending", "queued", "unschedulable")


def query_string(arguments) -> str:
    values = arguments.items() if isinstance(arguments, dict) else arguments
    pairs = [
        (key, str(value).lower() if isinstance(value, bool) else value)
        for key, value in values
        if value is not None and value != ""
    ]
    return "?" + urlencode(pairs) if pairs else ""


async def api_request(request, path: str, method="GET", body=None) -> dict:
    headers = {
        key: request.headers[key] for key in ("cookie", "authorization", "x-prism-csrf") if key in request.headers
    }
    headers.update(accept="application/json", **{"x-prism-source": "assistant"})
    principal = getattr(request.state, "principal", None)
    secret = get_settings().internal_auth_secret
    if principal and secret:
        timestamp = int(time.time())
        headers.update(
            {
                "x-prism-principal-id": principal.user_id,
                "x-prism-internal-ts": str(timestamp),
                "x-prism-internal-sig": sign_internal(
                    secret, timestamp=timestamp, method=method, path=urlsplit(path).path, principal_id=principal.user_id
                ),
            }
        )
    if request.headers.get("x-request-id"):
        headers["x-audit-parent-request-id"] = request.headers["x-request-id"]
    async with (
        asyncio.timeout(630),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=request.app, raise_app_exceptions=False),
            base_url=str(request.base_url),
            timeout=630,
        ) as client,
    ):
        response = await client.request(method, path, json=body, headers=headers)
    try:
        result = response.json() if response.content else None
    except ValueError:
        result = response.text
    return (
        result
        if response.is_success
        else {"error": f"upstream request failed (status {response.status_code})", "detail": result}
    )


def describe_state(result, keywords, *, nested=False):
    if not isinstance(result, dict):
        return None
    for key in STATUS_KEYS:
        value = result.get(key)
        if not nested and key == "ready" and value is True:
            return "ready=true"
        if isinstance(value, str) and any(word in value.lower() for word in keywords):
            return f"{key}={value}"
    if nested:
        for key in ("pods", "cases", "items"):
            for item in result.get(key, []) if isinstance(result.get(key), list) else []:
                if isinstance(item, dict):
                    for status_key in STATUS_KEYS:
                        value = item.get(status_key)
                        if isinstance(value, str) and any(word in value.lower() for word in keywords):
                            return f"{key}[].{status_key}={value}"
    return None


async def poll_until_terminal_or_pending(poll, timeout: float, interval: float) -> dict:
    started, attempts = time.monotonic(), 0
    while True:
        attempts += 1
        result = await poll()
        elapsed = math.floor(time.monotonic() - started + 0.5)
        base = {"attempts": attempts, "elapsedSeconds": elapsed, "result": result}
        if terminal := describe_state(result, TERMINAL):
            return {"done": True, "terminal": terminal, **base}
        if pending := describe_state(result, PENDING, nested=True):
            return {
                "done": False,
                "pending": True,
                **base,
                "message": (
                    f"Still pending/queued ({pending}) after {elapsed}s ({attempts} check(s)). "
                    "This usually means the cluster does not yet have free resources to schedule the "
                    "workload -- tell the user it looks resource-constrained and queued now instead of "
                    "waiting further. Call wait_for_status again with the same arguments only if they "
                    "want you to keep monitoring."
                ),
            }
        if time.monotonic() - started + interval > timeout:
            return {
                "done": False,
                **base,
                "message": (
                    f"Still not in a terminal state after {elapsed}s ({attempts} check(s)). "
                    "Call wait_for_status again with the same toolName/toolArguments to keep waiting."
                ),
            }
        await asyncio.sleep(interval)


def clean_arguments(value, schema):
    if isinstance(value, dict) and "properties" in schema:
        properties = schema["properties"]
        return {
            key: clean_arguments(item, properties.get(key, {}))
            for key, item in value.items()
            if key in properties or "additionalProperties" in schema
        }
    if isinstance(value, list):
        return [clean_arguments(item, schema.get("items", {})) for item in value]
    for alternative in schema.get("anyOf", []):
        if Draft202012Validator(alternative).is_valid(value):
            return clean_arguments(value, alternative)
    return value


async def invoke_tool(request, name: str, arguments: dict):
    tools = {tool["name"]: tool for tool in catalog(request.app)}
    tool = tools.get(name)
    if tool is None:
        raise ValueError(f'unknown tool "{name}"')
    try:
        Draft202012Validator(tool["inputSchema"]).validate(arguments)
    except ValidationError as error:
        location = ".".join(map(str, error.absolute_path)) or "arguments"
        raise ValueError(f"Invalid tool argument at {location}: {error.validator} constraint failed") from None
    arguments = clean_arguments(arguments, tool["inputSchema"])
    if "route" in tool:
        path = render_path(tool["route"], arguments) + query_string(
            {key: arguments.get(key) for key in tool["queryFields"]}
        )
        body = {key: arguments[key] for key in tool["bodyFields"] if key in arguments} or None
        return await api_request(request, path, tool["method"], body)
    if name in SPECIAL_ROUTES:
        method, path = SPECIAL_ROUTES[name]
        if method == "GET":
            return await api_request(request, path + query_string(arguments))
        return await api_request(request, path, method, arguments)
    if name == "get_storage_volume_resource_status":
        return await api_request(
            request,
            "/api/v1/storage/volume-resource-status"
            + query_string([("volume_id", value) for value in arguments["volumeIds"] if value]),
        )
    if name == "list_ready_deployments":
        result = await api_request(
            request,
            "/api/v1/deployments/executions"
            + query_string(
                {
                    "cluster_id": arguments.get("clusterId"),
                    "status": "ready",
                }
            ),
        )
        if result.get("error"):
            return {"deployments": [], "error": result["error"]}
        return {
            "deployments": [
                {
                    "id": str(item.get("execution_id") or ""),
                    "source": "deployment",
                    "namespace": item.get("namespace"),
                    "clusterId": item.get("cluster_id"),
                    "model": item.get("model"),
                    "ready": item.get("status") == "ready",
                    "gatewayEndpoint": item.get("endpoint") or item.get("forwarded_endpoint"),
                }
                for item in result.get("items", [])
            ]
        }
    if name == "search_aic_with_relaxation":
        result = await api_request(
            request, "/api/cluster/overview" + query_string({"clusterId": arguments["clusterId"]})
        )
        if result.get("error"):
            raise ValueError("Cluster overview failed: " + result["error"])
        hardware = result.get("hardware") or result.get("kubernetes", {}).get("hardware") or {}
        config = arguments.get("searchConfig") or {}
        mode = config.get("capacityMode", "free")
        total = hardware.get("gpuCount", 0)
        budget = (
            total
            if mode == "all"
            else config.get("customGpuCount")
            if mode == "custom"
            else hardware.get("availableGpuCount")
        )
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not float(budget).is_integer()
            or not 1 <= budget <= min(64, total)
        ):
            raise ValueError("Cluster overview has no valid AIC search GPU budget")
        return await api_request(
            request,
            "/api/candidate-search/relaxed",
            "POST",
            {
                "workload": arguments["workload"],
                "searchConfig": {**config, "totalGpus": budget},
                "sourceIds": ["aic"],
            },
        )
    if name == "wait_for_status":
        target_name = arguments["toolName"]
        target = tools.get(target_name)
        if target is None:
            return {"error": f'unknown tool "{target_name}"'}
        if target["riskTier"] != "read":
            return {
                "error": (
                    f'"{target_name}" is a {target["riskTier"]}-tier tool; '
                    "wait_for_status can only poll read-tier tools"
                )
            }
        return await poll_until_terminal_or_pending(
            lambda: invoke_tool(request, target_name, arguments.get("toolArguments") or {}),
            arguments.get("timeoutSeconds") or 60,
            arguments.get("intervalSeconds") or 5,
        )
    raise ValueError(f'No handler registered for "{name}"')
