"""Utilities for parsing human friendly time expressions."""

from __future__ import annotations


def parse_time(text: str) -> float:
    """Parse ``text`` as a time value in seconds.

    Accepted formats are:

    * ``HH:MM:SS``
    * ``MM:SS``
    * ``SS``

    Fractional seconds are supported.  ``ValueError`` is raised on
    malformed input.
    """

    parts = text.strip().split(":")
    if not parts:
        raise ValueError("empty time string")

    try:
        parts_f = [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("invalid time value") from exc

    if len(parts_f) == 1:
        seconds = parts_f[0]
    elif len(parts_f) == 2:
        minutes, seconds = parts_f
        seconds += minutes * 60
    elif len(parts_f) == 3:
        hours, minutes, seconds = parts_f
        seconds += minutes * 60 + hours * 3600
    else:
        raise ValueError("too many components in time string")
    return seconds
