"""Configuration structures for the :mod:`echopress.core` package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class CoreConfig:
    """Configuration options governing calibration and mapping.

    Attributes
    ----------
    alpha, beta:
        Calibration coefficients.
    scalar_channel:
        Index of the O-stream channel considered scalar.
    O_max:
        Maximum allowable alignment difference.
    tie_break:
        Tie-breaking strategy used during alignment.
    W:
        Window size when computing O-stream midpoints from boundaries.
    kappa:
        Multiplier applied to the alignment error metric.
    """

    alpha: Sequence[float] = field(default_factory=lambda: (1.0,))
    beta: Sequence[float] = field(default_factory=lambda: (0.0,))
    scalar_channel: int = 0
    O_max: float | None = None
    tie_break: str = "nearest"
    W: int = 1
    kappa: float = 1.0
