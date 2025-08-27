from __future__ import annotations

"""High level helpers for building datasets."""

from pathlib import Path
from typing import Sequence, Mapping, Any
import numpy as np

from .to_numpy import to_numpy


def from_arrays(
    features: Sequence[Sequence[float] | np.ndarray],
    target: Sequence[float] | np.ndarray,
    *,
    save_csv: str | Path | None = None,
    save_npz: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset directly from feature and target sequences."""
    return to_numpy(features, target, save_csv=save_csv, save_npz=save_npz)


def from_records(
    records: Sequence[Mapping[str, Any]],
    feature_keys: Sequence[str],
    target_key: str,
    *,
    save_csv: str | Path | None = None,
    save_npz: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset from a sequence of mapping objects.

    Each record in ``records`` must provide entries for all ``feature_keys`` and
    ``target_key``.  The values associated with ``feature_keys`` are collected
    in the same order to form the feature vectors.
    """
    feats = [[rec[k] for k in feature_keys] for rec in records]
    targ = [rec[target_key] for rec in records]
    return to_numpy(feats, targ, save_csv=save_csv, save_npz=save_npz)


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset previously saved via :func:`from_arrays`.

    Both ``.csv`` and ``.npz`` files are understood.  CSV files are assumed to
    contain feature columns followed by a single target column.
    """
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        return data["X"], data["y"]

    arr = np.loadtxt(p, delimiter=",")
    if arr.ndim == 1:
        # Handle single-sample edge case
        arr = arr[None, :]
    X = arr[:, :-1]
    y = arr[:, -1]
    return X, y
