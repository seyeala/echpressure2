from __future__ import annotations

"""Utilities for aligning O-stream and P-stream timelines."""

from dataclasses import dataclass, field
from typing import Sequence, Dict, Any

import numpy as np

from ..ingest import OStream, PStreamRecord
from ..config import Settings
from .derivative import central_difference
from .uncertainty import pressure_uncertainty


@dataclass
class AlignmentResult:
    """Result of aligning O- and P-streams at the file level.

    Attributes
    ----------
    mapping:
        Index of the P-stream record nearest to the file midpoint.
    E_align:
        Absolute time difference between the midpoint and the selected
        P-stream timestamp.
    diagnostics:
        Free-form dictionary containing any auxiliary information generated
        during the alignment process.
    """

    mapping: int
    E_align: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def align_streams(
    ostream: OStream,
    pstream: Sequence[PStreamRecord],
    *,
    settings: Settings | None = None,
    tie_breaker: str | None = None,
    O_max: float | None = None,
    W: int | None = None,
    kappa: float | None = None,
) -> AlignmentResult:
    """Align the O-stream file midpoint to the nearest P-stream timestamp.

    Parameters
    ----------
    ostream:
        :class:`OStream` containing sample timestamps ``T^O`` in seconds.
    pstream:
        Sequence of :class:`PStreamRecord` objects ordered by timestamp.
    settings:
        Optional :class:`~echopress.config.Settings` instance providing default
        values for the remaining parameters.
    tie_breaker, O_max, W, kappa:
        Individual overrides for the respective parameters.  Any value set here
        takes precedence over those supplied in ``settings``.

    Returns
    -------
    AlignmentResult
        Dataclass containing the index mapping, the alignment error and any
        diagnostics.
    """
    if settings is None:
        settings = Settings()

    tie_breaker = tie_breaker or settings.tie_breaker
    O_max = settings.O_max if O_max is None else O_max
    W = settings.W if W is None else W
    kappa = settings.kappa if kappa is None else kappa

    if tie_breaker not in {"earliest", "latest"}:
        raise ValueError("tie_breaker must be 'earliest' or 'latest'")

    o_times = np.asarray(ostream.timestamps, dtype=float)
    if o_times.ndim != 1 or o_times.size < 2:
        raise ValueError("ostream must contain at least two timestamps")

    midpoint = 0.5 * (o_times[0] + o_times[-1])

    p_times = np.array([rec.timestamp.timestamp() for rec in pstream], dtype=float)
    if p_times.size == 0:
        raise ValueError("pstream is empty")

    pressures = np.array([rec.pressure for rec in pstream], dtype=float)

    j = np.searchsorted(p_times, midpoint, side="left")
    if j == 0:
        mapping = 0
    elif j == len(p_times):
        mapping = len(p_times) - 1
    else:
        prev_diff = abs(midpoint - p_times[j - 1])
        next_diff = abs(p_times[j] - midpoint)
        if prev_diff < next_diff:
            mapping = j - 1
        elif prev_diff > next_diff:
            mapping = j
        else:
            mapping = j - 1 if tie_breaker == "earliest" else j
    E_align = abs(midpoint - p_times[mapping])
    if E_align > O_max:
        raise ValueError(
            f"Alignment error {E_align:.3f}s exceeds O_max"
        )

    # Derivative of the P-stream pressures
    if pressures.size >= 2:
        dt = float(np.mean(np.diff(p_times)))
    else:
        dt = 1.0
    W_eff = W
    if W_eff > pressures.size:
        W_eff = pressures.size - (1 - pressures.size % 2)
    if W_eff < 3:
        dp_dt_full = np.gradient(pressures, p_times, edge_order=1)
    else:
        dp_dt_full = central_difference(pressures, dt, W=W_eff)
    dp_dt = float(dp_dt_full[mapping])
    delta_p = pressure_uncertainty(dp_dt, E_align, kappa)

    diagnostics = {
        "tie_breaker": tie_breaker,
        "O_max": O_max,
        "W": W,
        "kappa": kappa,
        "midpoint": midpoint,
        "dp_dt": dp_dt,
        "delta_p": delta_p,
    }
    return AlignmentResult(mapping=mapping, E_align=E_align, diagnostics=diagnostics)

