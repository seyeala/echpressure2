"""Signal processing helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, List


def rms(data: Sequence[float]) -> float:
    """Return the root-mean-square of *data*.

    ``ValueError`` is raised for empty sequences.
    """

    if not data:
        raise ValueError("data must not be empty")
    return math.sqrt(sum(x * x for x in data) / len(data))


def moving_average(data: Sequence[float], window: int) -> List[float]:
    """Compute the simple moving average over *data*.

    The result has ``len(data) - window + 1`` elements.  ``ValueError`` is
    raised if ``window`` is not positive or greater than ``len(data)``.
    """

    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(data):
        raise ValueError("window larger than data")
    out: List[float] = []
    total = sum(data[:window])
    out.append(total / window)
    for i in range(window, len(data)):
        total += data[i] - data[i - window]
        out.append(total / window)
    return out
