"""Alignment algorithms for pressure and optical streams."""

from __future__ import annotations

from bisect import bisect_left
from math import sqrt
from typing import Iterable, List, Sequence, Tuple

from ..config import get_config, EchoPressConfig


Timestamp = float
Interval = Tuple[Timestamp, Timestamp]


def _resolve_tie(candidates: List[Tuple[float, int]], tie_break: str) -> int:
    """Resolve tie between candidates based on configuration."""
    candidates.sort(key=lambda x: (x[0], x[1]))
    min_dist = candidates[0][0]
    matches = [idx for dist, idx in candidates if abs(dist - min_dist) < 1e-12]
    if tie_break == "last":
        return matches[-1]
    # Default to "first" behaviour
    return matches[0]


def align_streams(
    p_times: Sequence[Timestamp],
    o_intervals: Sequence[Interval],
    *,
    config: EchoPressConfig | None = None,
) -> Tuple[List[int], float]:
    """Align an optical stream to a pressure stream.

    Parameters
    ----------
    p_times:
        Monotonically increasing timestamps for the pressure stream.
    o_intervals:
        Sequence of ``(start, end)`` tuples representing optical samples.
    config:
        Optional explicit configuration instance.  When omitted the
        module level configuration is used.

    Returns
    -------
    indices, E_align:
        ``indices`` contains, for each optical sample, the index of the
        nearest pressure timestamp.  ``E_align`` is the root mean square
        alignment error after applying the optional ``kappa`` multiplier.
    """

    cfg = config or get_config()
    if not p_times:
        raise ValueError("P-stream timestamps required")

    # Precompute P-stream for efficient searching
    p_times = list(p_times)
    o_mid = [(s + e) / 2 for s, e in o_intervals]

    indices: List[int] = []
    errors: List[float] = []
    prev_idx = 0
    for m in o_mid:
        search_lo = max(0, prev_idx - cfg.window_size)
        search_hi = min(len(p_times), prev_idx + cfg.window_size + 1)
        i = bisect_left(p_times, m, lo=search_lo, hi=search_hi)
        # Gather candidate neighbours
        cand: List[Tuple[float, int]] = []
        for idx in range(max(search_lo, i - 1), min(search_hi, i + 1)):
            cand.append((abs(p_times[idx] - m), idx))
        chosen = _resolve_tie(cand, cfg.tie_break)
        diff = p_times[chosen] - m
        if cfg.O_max is not None and abs(diff) > cfg.O_max:
            raise ValueError("O_max exceeded during alignment")
        indices.append(chosen)
        errors.append(diff)
        prev_idx = chosen

    if errors:
        mse = sum(e * e for e in errors) / len(errors)
        e_align = cfg.kappa * sqrt(mse)
    else:
        e_align = 0.0
    return indices, e_align
