from __future__ import annotations

"""Calibration utilities for converting measured voltages.

This module defines a small helper dataclass for storing per-channel
calibration coefficients as well as a convenience function to apply the
calibration to raw voltage measurements.

The calibration relationship for channel :math:`k` is

.. math::

   y_k = \alpha_k x + \beta_k

where ``x`` is the measured voltage and ``y_k`` is the calibrated value in the
application specific units.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationCoefficients:
    """Affine calibration coefficients for each acquisition channel.

    Parameters
    ----------
    alpha:
        Array of scale factors :math:`\alpha_k` for each channel. Dimensionless.
    beta:
        Array of offsets :math:`\beta_k` for each channel. Same units as the
        calibrated output.

    Notes
    -----
    The ``alpha`` and ``beta`` arrays must have identical shapes. Typically they
    are one-dimensional with length equal to the number of channels.
    """

    alpha: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)
        if self.alpha.shape != self.beta.shape:
            raise ValueError("alpha and beta must have identical shapes")


def apply_calibration(voltage: np.ndarray, coeffs: CalibrationCoefficients, channel: int) -> np.ndarray:
    """Apply affine calibration to a voltage trace for a specific channel.

    Parameters
    ----------
    voltage:
        Array of measured voltages in volts. ``voltage`` may have any shape and
        will be broadcast against the scalar coefficients. The calibration is
        applied element-wise.
    coeffs:
        :class:`CalibrationCoefficients` containing per-channel ``alpha`` and
        ``beta`` terms. ``coeffs.alpha[channel]`` and ``coeffs.beta[channel]``
        are used for the transformation.
    channel:
        Index of the channel whose coefficients should be applied. Must be
        compatible with ``coeffs``.

    Returns
    -------
    numpy.ndarray
        Calibrated values with the same shape as ``voltage``.

    Raises
    ------
    IndexError
        If ``channel`` is outside the valid range of the coefficient arrays.
    """

    alpha = coeffs.alpha[channel]
    beta = coeffs.beta[channel]
    return alpha * np.asarray(voltage) + beta
