"""Utility helpers for plotting and data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
_PLOT_WARNING_EMITTED = False
_HEADLESS_BACKENDS = {
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
}


def load_array(path: str | Path) -> np.ndarray:
    """Load a 1D numeric array from ``path``.

    ``.npy`` files are loaded with :func:`numpy.load` while any other extension
    is treated as a text file with comma separated values.
    """
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    return np.loadtxt(p, delimiter=",")


def auto_label(path: str | Path) -> str:
    """Return a concise label derived from ``path``.

    This helper is handy for the CLI entry points where a user often wants to
    plot a file quickly without manually specifying a label.  The file stem is
    generally descriptive enough and keeps the scripts terse.
    """
    return Path(path).stem


def plot_series(ax: plt.Axes, series: Sequence[float], label: str | None = None, **kwargs) -> None:
    """Plot a 1D series on ``ax`` with an optional label."""
    ax.plot(series, label=label, **kwargs)
    if label:
        ax.legend()


def _interactive_backend() -> bool:
    backend = matplotlib.get_backend()
    return backend in matplotlib.rcsetup.interactive_bk


def _headless_backend() -> bool:
    backend = matplotlib.get_backend()
    normalized = backend.lower()
    if normalized.startswith("module://"):
        normalized = normalized.split("module://", 1)[1]
    normalized = normalized.split(".")[-1]
    return normalized in _HEADLESS_BACKENDS


def _warn_on_headless_show() -> None:
    global _PLOT_WARNING_EMITTED
    if _PLOT_WARNING_EMITTED:
        return
    if _headless_backend():
        logger.warning(
            "Plotting uses plt.show() and may block in CLI/headless environments. "
            "Use --plot-save or --no-plot-show."
        )
        _PLOT_WARNING_EMITTED = True


def save_or_show(fig: plt.Figure, save: str | Path | None = None, show: bool = False) -> None:
    """Save ``fig`` to ``save`` or display it interactively.

    If ``save`` is ``None`` the figure will only be shown when ``show`` is
    True and the active backend is interactive.  This keeps non-interactive
    contexts from blocking while still allowing GUI-driven inspection.
    """
    if save:
        fig.savefig(save, bbox_inches="tight")
    if show:
        _warn_on_headless_show()
    if show and _interactive_backend():
        plt.show()
