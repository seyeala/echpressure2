"""Helpers for working with sliding windows over sequences."""

from __future__ import annotations

from typing import Iterator, Sequence, TypeVar, List

from ..types import Window

T = TypeVar("T")


def iter_windows(data: Sequence[T], size: int, step: int = 1) -> Iterator[Window]:
    """Yield ``Window`` objects describing slices of *data*.

    ``size`` is the window length and ``step`` controls how far the
    window advances each iteration.  ``ValueError`` is raised if the
    arguments are not sensible.
    """

    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    if size > len(data):
        raise ValueError("size larger than data")
    for start in range(0, len(data) - size + 1, step):
        yield Window(start, start + size)


def window_slices(data: Sequence[T], size: int, step: int = 1) -> List[Sequence[T]]:
    """Return the subsequences for each sliding window."""

    return [data[w.start : w.end] for w in iter_windows(data, size, step)]
