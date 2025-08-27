"""Uncertainty estimation utilities."""

from __future__ import annotations

import numpy as np


def pressure_uncertainty(dp_dt: np.ndarray | float, e_align: float, kappa: float) -> np.ndarray | float:
    """Compute the pressure uncertainty ΔP.

    Parameters
    ----------
    dp_dt:
        Time derivative of pressure.
    e_align:
        Alignment error factor.
    kappa:
        Proportionality constant relating |dp/dt| to pressure error.

    Returns
    -------
    numpy.ndarray or float
        Pressure uncertainty computed as ``kappa * abs(dp_dt) * e_align``.
    """

    return kappa * np.abs(dp_dt) * e_align


def bound_pressure(dp_dt: np.ndarray | float, e_align: float, kappa: float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return the ±ΔP bounds for pressure uncertainty.

    Parameters
    ----------
    dp_dt:
        Time derivative of pressure.
    e_align:
        Alignment error factor.
    kappa:
        Proportionality constant relating |dp/dt| to pressure error.

    Returns
    -------
    tuple of numpy.ndarray or float
        Pair ``(-ΔP, +ΔP)`` giving the negative and positive uncertainty bounds.
    """

    delta = pressure_uncertainty(dp_dt, e_align, kappa)
    return -delta, delta


__all__ = ["pressure_uncertainty", "bound_pressure"]

