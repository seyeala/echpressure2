"""Mapping utilities for aligning O-streams to P-streams.

Each O-stream file is assigned a scalar pressure label by aligning its
midpoint timestamp to the nearest P-stream timestamp.  The alignment
error ``E_align`` is reported and a maximum permissible misalignment
``O_max`` can be enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np


def _to_seconds(t: float | datetime) -> float:
    """Convert ``t`` to seconds since the Unix epoch."""
    if isinstance(t, datetime):
        return t.timestamp()
    return float(t)


@dataclass
class AlignmentResult:
    """Result of aligning an O-stream to the P-stream."""

    p_time: float | datetime
    pressure: float
    e_align: float


def align_midpoints(
    ostream_times: Sequence[float | datetime],
    p_times: Sequence[float | datetime],
    p_pressures: Sequence[float],
    O_max: float,
    tie_breaker: str = "earliest",
) -> AlignmentResult:
    """Align O-stream midpoint to nearest P-stream timestamp.

    Parameters
    ----------
    ostream_times:
        Timestamps of samples in the O-stream file.
    p_times:
        Sequence of P-stream timestamps.
    p_pressures:
        Pressure values corresponding to ``p_times``.
    O_max:
        Maximum acceptable alignment error in seconds.
    tie_breaker:
        Policy for resolving equidistant matches (``"earliest"`` or
        ``"latest"``).

    Returns
    -------
    AlignmentResult
        Dataclass containing the selected P-stream timestamp, its
        pressure and the alignment error ``E_align``.

    Raises
    ------
    ValueError
        If ``ostream_times`` is empty or ``E_align`` exceeds ``O_max``.
    """

    if len(ostream_times) == 0:
        raise ValueError("ostream_times must not be empty")

    t0 = _to_seconds(ostream_times[0])
    t1 = _to_seconds(ostream_times[-1])
    t_mid = (t0 + t1) / 2.0

    p_secs = np.array([_to_seconds(t) for t in p_times], dtype=float)
    diffs = np.abs(p_secs - t_mid)
    min_diff = diffs.min()
    candidate_idxs = np.flatnonzero(diffs == min_diff)
    if len(candidate_idxs) > 1:
        if tie_breaker == "latest":
            idx = candidate_idxs[-1]
        else:
            idx = candidate_idxs[0]
    else:
        idx = candidate_idxs[0]

    e_align = float(min_diff)
    if e_align > O_max:
        raise ValueError("Alignment error exceeds O_max")

    return AlignmentResult(
        p_time=p_times[idx], pressure=float(p_pressures[idx]), e_align=e_align
    )


__all__ = ["AlignmentResult", "align_midpoints"]
