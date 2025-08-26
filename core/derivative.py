"""Derivative estimation utilities.

This module provides several numerical estimators for the first
 derivative of a uniformly sampled time series.  All estimators operate
 on a configurable window ``W`` which represents the number of samples on
 either side of the evaluation point.  Edges where the estimator cannot
 be computed are filled with ``np.nan``.

The implementations favour clarity over performance which is adequate
 for small to medium sized time series used in unit tests.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np

ArrayLike = Iterable[float]


def _prepare_output(y: ArrayLike) -> np.ndarray:
    """Return an output array initialised with ``np.nan``.

    Parameters
    ----------
    y:
        Sequence of values representing the sampled signal.
    """
    y_arr = np.asarray(y, dtype=float)
    out = np.full_like(y_arr, np.nan, dtype=float)
    return y_arr, out


def central_difference(y: ArrayLike, dt: float = 1.0, W: int = 1) -> np.ndarray:
    """Estimate the derivative using a central difference scheme.

    The derivative at index ``i`` is computed as::

        (y[i + W] - y[i - W]) / (2 * W * dt)

    which reduces to the familiar three point stencil when ``W = 1``.
    ``np.nan`` is returned for indices where the computation is
    impossible (near the boundaries).
    """
    y_arr, out = _prepare_output(y)
    n = len(y_arr)
    if W < 1:
        raise ValueError("W must be at least 1")
    for i in range(W, n - W):
        out[i] = (y_arr[i + W] - y_arr[i - W]) / (2.0 * W * dt)
    return out


def local_linear(y: ArrayLike, dt: float = 1.0, W: int = 1) -> np.ndarray:
    """Estimate the derivative by fitting a local linear model.

    A straight line is fitted to the points in the window ``[i-W, i+W]``
    and the derivative is taken as the slope of that line.
    """
    y_arr, out = _prepare_output(y)
    n = len(y_arr)
    for i in range(n):
        start = max(0, i - W)
        end = min(n, i + W + 1)
        x = np.arange(start, end) * dt
        y_window = y_arr[start:end]
        if len(y_window) < 2:
            continue
        x_mean = x.mean()
        x_centered = x - x_mean
        coeffs = np.polyfit(x_centered, y_window, 1)
        out[i] = coeffs[0]
    return out


def savitzky_golay(
    y: ArrayLike, dt: float = 1.0, W: int = 1, poly_order: int = 3
) -> np.ndarray:
    """Savitzky–Golay derivative estimate.

    A polynomial of order ``poly_order`` is fitted to the samples in the
    window and differentiated analytically at the centre of the window.
    This implementation is intentionally lightweight and does not depend
    on :mod:`scipy`.
    """
    if poly_order < 1:
        raise ValueError("poly_order must be >= 1")
    y_arr, out = _prepare_output(y)
    n = len(y_arr)
    for i in range(n):
        start = max(0, i - W)
        end = min(n, i + W + 1)
        x = np.arange(start, end) * dt
        y_window = y_arr[start:end]
        if len(y_window) <= poly_order:
            continue
        x_center = x.mean()
        x_centered = x - x_center
        coeffs = np.polyfit(x_centered, y_window, poly_order)
        deriv_coeffs = np.polyder(coeffs)
        out[i] = np.polyval(deriv_coeffs, 0.0)
    return out

__all__ = ["central_difference", "local_linear", "savitzky_golay"]
