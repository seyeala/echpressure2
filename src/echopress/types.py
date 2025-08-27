"""Common type helpers for echopress.

This module defines lightweight containers and ``TypedDict`` instances
used across the codebase.  The structures are intentionally minimal but
add clarity and type safety around frequently exchanged data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict


class Sample(TypedDict):
    """Representation of a single data sample."""

    t: float
    value: float


@dataclass(frozen=True)
class TimeInterval:
    """Simple interval of time expressed in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Return the interval length in seconds."""

        return self.end - self.start


@dataclass(frozen=True)
class Window:
    """Index based window used for segmenting sequences."""

    start: int
    end: int

    @property
    def width(self) -> int:
        """Return the number of elements covered by the window."""

        return self.end - self.start


@dataclass
class TimeSeries:
    """Container for paired time and value sequences."""

    times: Sequence[float]
    values: Sequence[float]

    def __post_init__(self) -> None:  # pragma: no cover - tiny helper
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length")
