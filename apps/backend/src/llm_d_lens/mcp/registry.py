"""Tool registration, argument validation and dispatch."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .context import ToolContext
from .errors import ToolError
from .lens_api import LensApiError
from .safety import find_secret, redact

logger = logging.getLogger(__name__)

PROFILES = ("core", "serving", "storage", "simulation", "cluster-admin", "lens-settings")

Handler = Callable[[ToolContext, Any], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    read_only: bool
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False
    profile: str = "core"

    def input_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema


_TOOLS: dict[str, ToolSpec] = {}


def tool(
    name: str,
    description: str,
    input_model: type[BaseModel],
    *,
    read_only: bool,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
    profile: str = "core",
):
    """Register a handler. Annotations describe real behaviour so the host can apply its own approval policy."""

    def register(handler: Handler) -> Handler:
        if name in _TOOLS:
            raise ValueError(f"duplicate tool {name}")
        _TOOLS[name] = ToolSpec(
            name,
            description,
            input_model,
            handler,
            read_only,
            destructive,
            idempotent or read_only,
            open_world,
            profile,
        )
        return handler

    return register


def all_tools(scopes: frozenset[str] | set[str] | None = None) -> list[ToolSpec]:
    from . import tools  # noqa: F401  (importing registers every tool)

    return [spec for spec in _TOOLS.values() if scopes is None or spec.profile in scopes]


def _validation_error(error: ValidationError) -> ToolError:
    problems = [(".".join(str(part) for part in item["loc"]) or None, item["msg"]) for item in error.errors()]
    field, message = problems[0]
    extra = {"all_problems": [{"field": name, "message": text} for name, text in problems]} if len(problems) > 1 else {}
    return ToolError(
        "invalid_argument",
        f"{field or 'arguments'}: {message}" + (f" ({len(problems)} problems; see all_problems)" if extra else ""),
        field=field,
        hint="Check the tool's input schema and retry with corrected arguments.",
        **extra,
    )


def upstream_error(error: LensApiError) -> ToolError:
    hints = {
        401: "Lens rejected the credentials; the session may have expired.",
        403: "The Lens user lacks permission for this operation.",
        404: "Lens could not find the resource.",
        409: "The resource is in a state that conflicts with this operation.",
    }
    code = error.code() or f"lens_{error.status}"
    hint = hints.get(error.status)
    if hint is None and error.status >= 500:
        hint = (
            "Lens failed on the server side and does not expose the cause to clients. The operation may not have "
            "completed: check resource_list before retrying, and ask the Lens administrator to read the server log."
        )
    if code == "cluster_not_found":
        hint = "Call cluster_list."
    return ToolError(
        code,
        error.message(),
        hint=hint,
        upstream_status=error.status,
    )


async def call_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Run a tool and return ``(payload, is_error)``; never raises for expected failures."""
    payload, is_error = await _dispatch(ctx, name, arguments)
    return redact(payload), is_error


async def _dispatch(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    spec = next((item for item in all_tools() if item.name == name), None)
    if spec is None:
        return ToolError(
            "unknown_tool", f"No tool named {name}", hint="List tools and pick an existing name."
        ).payload(), True
    if spec.profile not in ctx.scopes:
        return ToolError(
            "scope_required",
            f"{name} belongs to the '{spec.profile}' profile, which this connection does not have.",
            hint="This connection cannot perform the operation. Ask the user to start the server with that profile "
            "(LENS_MCP_PROFILES) if it is really needed.",
        ).payload(), True
    try:
        args = spec.input_model.model_validate(arguments or {})
        secret = find_secret(arguments or {})
        if secret:
            raise ToolError(
                "secret_in_argument",
                f"{secret} looks like an access token or secret.",
                field=secret.split(" ")[0],
                hint="Lens does not accept plaintext secrets. Use the credentials already configured in the cluster.",
            )
        return await spec.handler(ctx, args), False
    except ValidationError as error:
        failure = _validation_error(error)
    except ToolError as error:
        failure = error
    except LensApiError as error:
        failure = upstream_error(error)
        if failure.code == "cluster_not_found" and arguments.get("cluster_id"):
            failure = ToolError(
                "not_found",
                f"No cluster {arguments['cluster_id']}.",
                field="cluster_id",
                hint=failure.hint,
            )
    except Exception as error:  # noqa: BLE001 - never leak an unstructured failure to the agent
        logger.exception("MCP tool %s failed", name)
        failure = ToolError(
            "upstream_unavailable" if isinstance(error, (TimeoutError, OSError)) else "internal_error",
            f"The tool failed unexpectedly ({type(error).__name__}); the cause is in the Lens log.",
            hint="The operation may or may not have completed. Re-check with the matching get_* tool before retrying.",
        )
    return failure.payload(), True
