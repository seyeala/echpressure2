"""Numerical derivative estimation utilities.

This module provides windowed derivative estimators for one-dimensional
series data.  All routines expect a sampling period ``dt`` and an odd
window length ``W``.  Points near the array boundaries are handled using
forward or backward schemes so that an estimate is produced for every
sample.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter

from ..config import Settings


def _validate_window(W: int, n: int) -> None:
    """Validate window size ``W`` against ``n`` samples."""
    if W % 2 == 0:
        raise ValueError("W must be an odd integer")
    if W < 1:
        raise ValueError("W must be positive")
    if W > n:
        raise ValueError("W must not exceed the length of the series")


def central_difference(
    series: Sequence[float],
    dt: float,
    W: int | None = None,
    *,
    settings: Settings | None = None,
    kappa: float | None = None,
) -> np.ndarray:
    """Estimate the first derivative using a central difference scheme.

    For interior points a symmetric ``\u00b1W//2`` window is used.  For samples
    within ``W//2`` points of either boundary a purely forward or backward
    difference of span ``W-1`` is applied.

    Parameters
    ----------
    series:
        Input sample values.
    dt:
        Sampling period between successive samples.
    W:
        Odd window length.  ``W`` must not exceed the number of samples in
        ``series``.

    Returns
    -------
    numpy.ndarray
        Array of derivative estimates matching the length of ``series``.
    """

    if settings is None:
        settings = Settings()

    if W is None:
        W = settings.mapping.W
    if kappa is None:
        kappa = settings.mapping.kappa  # currently unused but kept for API symmetry

    arr = np.asarray(series, dtype=float)
    n = arr.size
    _validate_window(W, n)
    half = W // 2
    out = np.empty(n, dtype=float)
    span = (W - 1) * dt
    for i in range(n):
        if i < half:
            out[i] = (arr[i + W - 1] - arr[i]) / span
        elif i >= n - half:
            out[i] = (arr[i] - arr[i - W + 1]) / span
        else:
            out[i] = (arr[i + half] - arr[i - half]) / span
    return out


def local_linear(
    series: Sequence[float],
    dt: float,
    W: int | None = None,
    *,
    settings: Settings | None = None,
    kappa: float | None = None,
) -> np.ndarray:
    """Estimate derivatives via local linear regression.

    A first-order polynomial is fit to each window of ``W`` consecutive
    samples and the slope at the center of the window is returned.  Near the
    boundaries the window is shifted forward or backward to use the available
    data.

    Parameters
    ----------
    series:
        Input sample values.
    dt:
        Sampling period between successive samples.
    W:
        Odd window length.  ``W`` must not exceed the number of samples in
        ``series``.

    Returns
    -------
    numpy.ndarray
        Array of derivative estimates matching the length of ``series``.
    """

    if settings is None:
        settings = Settings()

    if W is None:
        W = settings.mapping.W
    if kappa is None:
        kappa = settings.mapping.kappa  # unused

    arr = np.asarray(series, dtype=float)
    n = arr.size
    _validate_window(W, n)
    half = W // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        start = i - half
        if start < 0:
            start = 0
        elif start > n - W:
            start = n - W
        end = start + W
        x = dt * np.arange(start, end)
        y = arr[start:end]
        slope, _ = np.polyfit(x, y, 1)
        out[i] = slope
    return out


def savgol(
    series: Sequence[float],
    dt: float,
    W: int | None = None,
    *,
    settings: Settings | None = None,
    kappa: float | None = None,
    polyorder: int = 2,
) -> np.ndarray:
    """Estimate derivatives using a Savitzky\u2013Golay filter.

    The filter performs a polynomial least-squares fit over each window and
    returns the first derivative.  ``W`` must be an odd integer greater than
    ``polyorder``.  Edge samples are computed using SciPy's ``interp`` mode
    which fits polynomials to the boundary points.

    Parameters
    ----------
    series:
        Input sample values.
    dt:
        Sampling period between successive samples.
    W:
        Odd window length.  ``W`` must not exceed the number of samples in
        ``series``.
    polyorder:
        Order of the polynomial used in the local fit.  The default is 2.

    Returns
    -------
    numpy.ndarray
        Array of derivative estimates matching the length of ``series``.
    """

    if settings is None:
        settings = Settings()

    if W is None:
        W = settings.mapping.W
    if kappa is None:
        kappa = settings.mapping.kappa  # unused

    arr = np.asarray(series, dtype=float)
    n = arr.size
    _validate_window(W, n)
    if polyorder >= W:
        raise ValueError("polyorder must be less than W")
    return savgol_filter(arr, window_length=W, polyorder=polyorder, deriv=1, delta=dt, mode="interp")
