"""Uncertainty computations for pressure estimates.

The relationship between pressure change :math:`\Delta P`, the time
 derivative of pressure :math:`\frac{dp}{dt}` and a normalised alignment
 error :math:`E_{align}` is approximated as::

    |ΔP| ≤ κ |dp/dt| · E_align

This module exposes small helper utilities to compute the bound and to
 verify whether an observed change satisfies it.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def pressure_bound(dp_dt: Iterable[float] | float, e_align: float, kappa: float = 1.0) -> np.ndarray:
    """Return the uncertainty bound ``κ |dp/dt| * E_align``.

    Parameters
    ----------
    dp_dt:
        Derivative of pressure with respect to time.  May be a scalar or a
        sequence.
    e_align:
        Alignment error factor ``E_align``.
    kappa:
        Empirical scaling constant ``κ``.
    """
    dp_dt_arr = np.asarray(dp_dt, dtype=float)
    return kappa * np.abs(dp_dt_arr) * float(e_align)


def within_bound(delta_p: Iterable[float] | float, dp_dt: Iterable[float] | float,
                 e_align: float, kappa: float = 1.0) -> np.ndarray:
    """Check whether ``|ΔP|`` is within the computed bound.

    Parameters
    ----------
    delta_p:
        Observed pressure change ``ΔP``.
    dp_dt:
        Derivative of pressure ``dp/dt``.
    e_align:
        Alignment error ``E_align``.
    kappa:
        Empirical scaling constant ``κ``.
    """
    delta_p_arr = np.asarray(delta_p, dtype=float)
    bound = pressure_bound(dp_dt, e_align, kappa)
    return np.abs(delta_p_arr) <= bound

__all__ = ["pressure_bound", "within_bound"]
