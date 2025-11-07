"""Compatibility helpers for Typer exceptions."""

from __future__ import annotations

from typing import Any, NoReturn, Optional

import typer


def bad_parameter(
    message: str,
    *,
    ctx: Optional[typer.Context] = None,
    param: Any = None,
    param_hint: Optional[str] = None,
    param_name: Optional[str] = None,
) -> NoReturn:
    """Raise :class:`typer.BadParameter` with backwards-compatible hints.

    Typer 0.9 removed the ``param_name`` keyword argument that older code
    relied on.  This helper accepts both ``param_name`` and ``param_hint`` and
    forwards the information using the supported ``param_hint`` parameter.  Any
    provided ``ctx`` or ``param`` objects are passed through unchanged.
    """

    kwargs: dict[str, Any] = {}
    if ctx is not None:
        kwargs["ctx"] = ctx
    if param is not None:
        kwargs["param"] = param
    hint = param_hint or param_name
    if hint is not None:
        kwargs["param_hint"] = hint
    raise typer.BadParameter(message, **kwargs)

