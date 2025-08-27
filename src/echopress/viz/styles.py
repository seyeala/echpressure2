"""Matplotlib styles for echpressure2 visualisations."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Base style configuration used across all plots.  The values can be
# overridden by supplying a different style mapping to :func:`apply_style`.
BASE_STYLE = {
    "figure.figsize": (10, 4),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.titlesize": "large",
    "axes.labelsize": "medium",
    "lines.linewidth": 1.5,
}


def apply_style(extra: dict | None = None) -> None:
    """Apply a consistent matplotlib style.

    Parameters
    ----------
    extra:
        Optional dictionary of rcParams that override the base style.
    """
    style = BASE_STYLE.copy()
    if extra:
        style.update(extra)
    plt.rcParams.update(style)
