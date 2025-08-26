"""Utility helpers for plotting and data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_array(path: str | Path) -> np.ndarray:
    """Load a 1D numeric array from ``path``.

    ``.npy`` files are loaded with :func:`numpy.load` while any other extension
    is treated as a text file with comma separated values.
    """
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    return np.loadtxt(p, delimiter=",")


def plot_series(ax: plt.Axes, series: Sequence[float], label: str | None = None, **kwargs) -> None:
    """Plot a 1D series on ``ax`` with an optional label."""
    ax.plot(series, label=label, **kwargs)
    if label:
        ax.legend()


def save_or_show(fig: plt.Figure, save: str | Path | None = None, show: bool = False) -> None:
    """Save ``fig`` to ``save`` or display it interactively.

    If ``save`` is ``None`` the figure will only be shown when ``show`` is
    True.  When both are unset the figure is shown by default to give quick
    feedback during inspection.
    """
    if save:
        fig.savefig(save, bbox_inches="tight")
    if show or not save:
        plt.show()
