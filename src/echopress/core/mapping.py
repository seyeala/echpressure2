from __future__ import annotations

"""Utilities for aligning O-stream and P-stream timelines."""

from dataclasses import dataclass, field
from typing import Sequence, Dict, Any

import numpy as np

from ..ingest import OStream, PStreamRecord


@dataclass
class AlignmentResult:
    """Result of aligning O- and P-streams.

    Attributes
    ----------
    mapping:
        Array mapping each O-stream midpoint to the index of the nearest
        P-stream record.
    E_align:
        Array of absolute time differences between each midpoint and the
        selected P-stream timestamp.
    diagnostics:
        Free-form dictionary containing any auxiliary information generated
        during the alignment process.
    """

    mapping: np.ndarray
    E_align: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def align_streams(
    ostream: OStream,
    pstream: Sequence[PStreamRecord],
    *,
    tie_break: str,
    O_max: float,
    W: int,
    kappa: float,
) -> AlignmentResult:
    """Align O-stream sample midpoints to the nearest P-stream timestamps.

    Parameters
    ----------
    ostream:
        :class:`OStream` containing sample timestamps ``T^O`` in seconds.
    pstream:
        Sequence of :class:`PStreamRecord` objects ordered by timestamp.
    tie_break:
        Strategy used when a midpoint is equidistant from two P-stream
        timestamps.  ``"earlier"`` selects the earlier record whereas
        ``"later"`` selects the later one.
    O_max:
        Maximum permissible alignment error in seconds.  If any midpoint is
        farther than ``O_max`` from the nearest P-stream record a
        :class:`ValueError` is raised.
    W, kappa:
        Currently unused but retained for diagnostic purposes.

    Returns
    -------
    AlignmentResult
        Dataclass containing the index mapping, the per-midpoint alignment
        error array and any diagnostics.
    """
    if tie_break not in {"earlier", "later"}:
        raise ValueError("tie_break must be 'earlier' or 'later'")

    o_times = np.asarray(ostream.timestamps, dtype=float)
    if o_times.ndim != 1 or o_times.size < 2:
        raise ValueError("ostream must contain at least two timestamps")

    midpoints = (o_times[:-1] + o_times[1:]) / 2.0

    p_times = np.array([rec.timestamp.timestamp() for rec in pstream], dtype=float)
    if p_times.size == 0:
        raise ValueError("pstream is empty")

    mapping = np.empty(midpoints.shape, dtype=int)
    E_align = np.empty(midpoints.shape, dtype=float)

    for i, mp in enumerate(midpoints):
        j = np.searchsorted(p_times, mp, side="left")
        if j == 0:
            idx = 0
        elif j == len(p_times):
            idx = len(p_times) - 1
        else:
            prev_diff = abs(mp - p_times[j - 1])
            next_diff = abs(p_times[j] - mp)
            if prev_diff < next_diff:
                idx = j - 1
            elif prev_diff > next_diff:
                idx = j
            else:
                idx = j - 1 if tie_break == "earlier" else j
        mapping[i] = idx
        E_align[i] = abs(mp - p_times[idx])
        if E_align[i] > O_max:
            raise ValueError(
                f"Alignment error {E_align[i]:.3f}s exceeds O_max at index {i}"
            )

    diagnostics = {"tie_break": tie_break, "O_max": O_max, "W": W, "kappa": kappa, "midpoints": midpoints}
    return AlignmentResult(mapping=mapping, E_align=E_align, diagnostics=diagnostics)

