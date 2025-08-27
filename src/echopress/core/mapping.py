"""Temporal mapping utilities between O- and P-streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class AlignmentResult:
    """Result of aligning an O-stream to a P-stream."""

    indices: np.ndarray
    e_align: float
    midpoints: np.ndarray
    deltas: np.ndarray


# ----------------------------------------------------------------------
def _midpoints(timestamps: np.ndarray, window: int) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=float)
    if ts.ndim == 2 and ts.shape[1] == 2:
        return ts.mean(axis=1)
    if ts.ndim == 1:
        if len(ts) < window + 1:
            raise ValueError("not enough timestamps for window")
        return (ts[:-window] + ts[window:]) / 2.0
    raise ValueError("Unsupported timestamps shape")


# ----------------------------------------------------------------------
def align_midpoints(
    p_times: Sequence[float],
    o_times: Sequence[float] | np.ndarray,
    *,
    O_max: float | None = None,
    tie_break: str = "nearest",
    W: int = 1,
    kappa: float = 1.0,
) -> AlignmentResult:
    """Align O-stream midpoints to the nearest P-stream timestamps.

    Parameters
    ----------
    p_times:
        Sequence of P-stream timestamps.
    o_times:
        O-stream timestamps; either ``(N, 2)`` start/end pairs or a one-
        dimensional array of boundaries.  Midpoints are computed using a
        window of size ``W`` when ``o_times`` is one-dimensional.
    O_max:
        Optional maximum allowable absolute alignment difference.  If any
        matched pair exceeds this threshold a :class:`ValueError` is raised.
    tie_break:
        Strategy used when an O midpoint is equidistant to two P timestamps.
        ``"earlier"`` chooses the preceding timestamp, ``"later"`` chooses
        the following timestamp and ``"nearest"`` selects the earlier one by
        default.
    W:
        Window size used when computing midpoints from one-dimensional
        ``o_times`` arrays.
    kappa:
        Multiplier applied to the alignment error metric ``E_align``.
    """

    p = np.asarray(p_times, dtype=float)
    mid = _midpoints(np.asarray(o_times, dtype=float), W)

    idx = np.searchsorted(p, mid)
    prev_idx = np.clip(idx - 1, 0, len(p) - 1)
    next_idx = np.clip(idx, 0, len(p) - 1)

    prev_delta = np.abs(mid - p[prev_idx])
    next_delta = np.abs(mid - p[next_idx])

    if tie_break == "earlier":
        choose_prev = prev_delta <= next_delta
    elif tie_break == "later":
        choose_prev = prev_delta < next_delta
    else:  # default: nearest
        choose_prev = prev_delta <= next_delta

    mapping = np.where(choose_prev, prev_idx, next_idx)
    deltas = np.abs(mid - p[mapping])

    if O_max is not None and np.any(deltas > O_max):
        raise ValueError("alignment exceeds O_max")

    e_align = kappa * float(np.sum(deltas))
    return AlignmentResult(indices=mapping, e_align=e_align, midpoints=mid, deltas=deltas)
