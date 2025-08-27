"""Voltage-to-pressure calibration utilities.

This module implements a simple affine calibration model mapping
voltages ``v_k`` to pressures ``p_k`` using per-channel coefficients
``alpha_k`` and ``beta_k``::

    p_k = alpha_k * v_k + beta_k

The :func:`calibrate` function handles either single samples or arrays of
samples.  An optional ``channel`` argument can be supplied to return a
specific calibrated channel.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def calibrate(
    voltages: Sequence[float] | np.ndarray,
    alpha: Sequence[float],
    beta: Sequence[float],
    channel: int | None = None,
) -> np.ndarray | float:
    """Convert voltages to pressure values.

    Parameters
    ----------
    voltages:
        Voltage measurements.  The last dimension corresponds to channels.
    alpha:
        Per-channel slope coefficients.
    beta:
        Per-channel intercept coefficients.
    channel:
        Optional channel index to select a single calibrated channel.

    Returns
    -------
    numpy.ndarray or float
        Calibrated pressure values.  If ``channel`` is provided, a scalar
        pressure for that channel is returned.
    """

    v = np.asarray(voltages, dtype=float)
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    pressures = a * v + b
    if channel is not None:
        return pressures[..., channel]
    return pressures


__all__ = ["calibrate"]
