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

from ..config import Settings


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


def apply_calibration(
    voltage: np.ndarray,
    coeffs: CalibrationCoefficients | None = None,
    channel: int | None = None,
    *,
    settings: Settings | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> np.ndarray:
    """Apply affine calibration to a voltage trace for a specific channel.

    Parameters
    ----------
    voltage:
        Array of measured voltages in volts. ``voltage`` may have any shape and
        will be broadcast against the scalar coefficients. The calibration is
        applied element-wise.
    coeffs:
        Optional :class:`CalibrationCoefficients` containing per-channel
        ``alpha`` and ``beta`` terms.  If provided these take precedence over
        values supplied via ``settings`` or explicit ``alpha``/``beta``
        arguments.
    channel:
        Index of the channel whose coefficients should be applied. If ``None``
        the value is taken from ``settings.pressure.scalar_channel``.
    settings:
        Optional :class:`~echopress.config.Settings` instance providing default
        values.
    alpha, beta:
        Scalar calibration coefficients.  These override any corresponding
        values drawn from ``coeffs`` or ``settings``.

    Returns
    -------
    numpy.ndarray
        Calibrated values with the same shape as ``voltage``.
    """

    if settings is None:
        settings = Settings()

    if channel is None:
        channel = settings.pressure.scalar_channel

    if coeffs is not None:
        alpha = coeffs.alpha[channel]
        beta = coeffs.beta[channel]
    else:
        if alpha is None:
            al = settings.calibration.alpha
            alpha = al[channel] if channel < len(al) else al[0]
        if beta is None:
            be = settings.calibration.beta
            beta = be[channel] if channel < len(be) else be[0]

    if alpha is None or beta is None:
        raise ValueError("alpha and beta coefficients must be specified")

    return alpha * np.asarray(voltage) + beta
