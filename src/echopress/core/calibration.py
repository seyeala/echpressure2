"""Voltage-to-pressure calibration utilities.

The module exposes :class:`Calibrator` for encapsulating calibration
coefficients and a convenience :func:`calibrate` function.  Each channel ``k``
uses coefficients :math:`\alpha_k` and :math:`\beta_k` such that::

    p_k = \alpha_k v_k + \beta_k

where ``v_k`` is the measured voltage and ``p_k`` is the estimated pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class Calibrator:
    """Simple affine calibrator using ``alpha`` and ``beta`` coefficients.

    Parameters
    ----------
    alpha, beta:
        Sequences of per-channel coefficients.  They must have the same
        length.  ``alpha`` corresponds to the multiplicative term and
        ``beta`` to the additive term.
    """

    alpha: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)
        if self.alpha.shape != self.beta.shape:
            raise ValueError("alpha and beta must have the same shape")

    # ------------------------------------------------------------------
    def __call__(self, voltages: np.ndarray, channel: int | None = None) -> np.ndarray:
        """Map ``voltages`` to pressures.

        Parameters
        ----------
        voltages:
            Array of voltage readings.  The last dimension is interpreted as
            channel index.
        channel:
            Optional channel selection.  When provided only that channel is
            processed and a one-dimensional array is returned.
        """
        v = np.asarray(voltages, dtype=float)
        if channel is not None:
            return self.alpha[channel] * v[..., channel] + self.beta[channel]
        return self.alpha * v + self.beta


# ----------------------------------------------------------------------
def calibrate(
    voltages: np.ndarray,
    alpha: Sequence[float],
    beta: Sequence[float],
    *,
    channel: int | None = None,
) -> np.ndarray:
    """Calibrate ``voltages`` using per-channel ``alpha`` and ``beta``.

    This is a thin wrapper around :class:`Calibrator` for convenience.
    """
    return Calibrator(np.asarray(alpha), np.asarray(beta))(voltages, channel=channel)
