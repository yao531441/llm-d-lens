"""Tool registration, argument validation and dispatch."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .context import ToolContext
from .errors import ToolError
from .lens_api import LensApiError
from .safety import find_secret, redact

PROFILES = ("core", "serving", "storage", "simulation", "cluster-admin", "lens-settings")
DESTRUCTIVE_PER_HOUR = int(os.environ.get("LENS_MCP_V2_DESTRUCTIVE_PER_HOUR", "100"))

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
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or None
    return ToolError(
        "invalid_argument",
        f"{field or 'arguments'}: {first['msg']}",
        field=field,
        hint="Check the tool's input schema and retry with corrected arguments.",
    )


def upstream_error(error: LensApiError) -> ToolError:
    hints = {
        401: "Lens rejected the credentials; the session may have expired.",
        403: "The Lens user lacks permission for this operation.",
        404: "Lens could not find the resource.",
        409: "The resource is in a state that conflicts with this operation.",
    }
    return ToolError(
        error.code() or f"lens_{error.status}",
        error.message(),
        hint=hints.get(error.status),
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
            "(LENS_MCP_V2_PROFILES) if it is really needed.",
        ).payload(), True
    replay_key = None
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
        if not spec.read_only:
            replay_key = getattr(args, "client_request_id", None)
            if replay_key:
                cached = ctx.ledger.get_idempotent(ctx.owner, name, replay_key)
                if cached is not None:
                    return {**cached, "meta": {"idempotent_replay": True}}, False
            if spec.destructive and ctx.ledger.count_actions(ctx.owner, "destructive") >= DESTRUCTIVE_PER_HOUR:
                raise ToolError(
                    "rate_limited",
                    f"{DESTRUCTIVE_PER_HOUR} destructive operations were already performed in the last hour.",
                    hint="Wait, or ask the user to do bulk cleanup in the Lens UI.",
                )
            if spec.destructive:
                ctx.ledger.record_action(ctx.owner, "destructive")
        result = await spec.handler(ctx, args)
        if replay_key:
            ctx.ledger.put_idempotent(ctx.owner, name, replay_key, result)
        _audit(ctx, spec, arguments, False, None)
        return result, False
    except ValidationError as error:
        failure = _validation_error(error)
    except ToolError as error:
        failure = error
    except LensApiError as error:
        failure = upstream_error(error)
    _audit(ctx, spec, arguments, True, failure.message)
    return failure.payload(), True


def _audit(ctx: ToolContext, spec: ToolSpec, arguments: dict[str, Any], is_error: bool, summary: str | None) -> None:
    if not spec.read_only:
        ctx.ledger.audit(ctx.owner, spec.name, redact(arguments or {}), is_error, summary)
