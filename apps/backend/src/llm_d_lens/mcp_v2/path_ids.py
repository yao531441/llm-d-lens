"""Ids that are interpolated into Lens URL paths must be single path segments."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints

from .errors import ToolError

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
_ID = re.compile(ID_PATTERN)
PathId = Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=ID_PATTERN)]


def is_path_id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 100 and _ID.fullmatch(value) is not None


def check_id(value: str, field: str) -> str:
    if not is_path_id(value):
        raise ToolError(
            "invalid_argument",
            f"'{value[:60]}' is not a valid id.",
            field=field,
            hint="Ids contain only letters, digits, '_', '.' and '-'.",
        )
    return value
