from __future__ import annotations

"""Utilities for converting feature/label collections into NumPy arrays."""

from pathlib import Path
from typing import Sequence, Iterable, Mapping, Any
import numpy as np


def to_numpy(
    features: Sequence[Sequence[float] | np.ndarray],
    target: Sequence[float] | np.ndarray,
    *,
    save_csv: str | Path | None = None,
    save_npz: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` NumPy arrays from in-memory sequences.

    Parameters
    ----------
    features:
        Sequence of feature vectors.  Each element is converted to a 1-D array
        and stacked to form ``X`` with shape ``(n_samples, n_features)``.
    target:
        Sequence of target values with length ``n_samples``.
    save_csv, save_npz:
        Optional paths.  If provided the dataset is persisted either as a CSV
        file (features followed by target column) or an ``.npz`` archive with
        ``X`` and ``y`` entries.
    """

    X = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Features and target must contain the same number of samples")

    if save_csv:
        path = Path(save_csv)
        arr = np.hstack([X, y[:, None]])
        np.savetxt(path, arr, delimiter=",")

    if save_npz:
        path = Path(save_npz)
        np.savez(path, X=X, y=y)

    return X, y
