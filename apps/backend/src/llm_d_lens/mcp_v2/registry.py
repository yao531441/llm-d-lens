"""Tool registration, argument validation and dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .context import ToolContext
from .errors import ToolError
from .lens_api import LensApiError

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
):
    """Register a handler. Annotations describe real behaviour so the host can apply its own approval policy."""

    def register(handler: Handler) -> Handler:
        if name in _TOOLS:
            raise ValueError(f"duplicate tool {name}")
        _TOOLS[name] = ToolSpec(
            name, description, input_model, handler, read_only, destructive, idempotent or read_only, open_world
        )
        return handler

    return register


def all_tools() -> list[ToolSpec]:
    from . import tools  # noqa: F401  (importing registers every tool)

    return list(_TOOLS.values())


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
    spec = next((item for item in all_tools() if item.name == name), None)
    if spec is None:
        return ToolError(
            "unknown_tool", f"No tool named {name}", hint="List tools and pick an existing name."
        ).payload(), True
    try:
        args = spec.input_model.model_validate(arguments or {})
        return await spec.handler(ctx, args), False
    except ValidationError as error:
        return _validation_error(error).payload(), True
    except ToolError as error:
        return error.payload(), True
    except LensApiError as error:
        return upstream_error(error).payload(), True
