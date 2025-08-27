"""Utilities for converting datasets to NumPy arrays.

This module provides helper functions to split a pandas ``DataFrame`` into
feature and target ``numpy.ndarray`` objects.  The resulting arrays can be
optionally persisted to disk in either CSV or NPZ formats.  The functions are
intended to be lightweight building blocks for experiments and quick data
ingestion routines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def dataframe_to_numpy(
    df: pd.DataFrame,
    target: str,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    save_csv: Optional[Path] = None,
    save_npz: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a :class:`~pandas.DataFrame` into ``(X, y)`` arrays.

    Parameters
    ----------
    df:
        Input dataframe containing both features and a target column.
    target:
        Name of the column to use as the target ``y``.
    feature_columns:
        Optional explicit list of feature columns.  When ``None`` all columns
        except ``target`` are treated as features.
    save_csv:
        When provided, the full dataframe is written to this path using
        :meth:`pandas.DataFrame.to_csv`.
    save_npz:
        When provided, a compressed ``npz`` archive containing ``X`` and ``y``
        is written to this path using :func:`numpy.savez_compressed`.

    Returns
    -------
    tuple of ``numpy.ndarray``
        A tuple ``(X, y)`` where ``X`` contains the feature matrix and ``y``
        contains the target vector.
    """

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target]

    # Extract features and target as numpy arrays
    X = df[feature_columns].to_numpy()
    y = df[target].to_numpy()

    if save_csv is not None:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    if save_npz is not None:
        Path(save_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_npz, X=X, y=y)

    return X, y


def arrays_to_npz(
    X: np.ndarray,
    y: np.ndarray,
    path: Path,
) -> Path:
    """Persist ``X`` and ``y`` arrays to an ``npz`` file.

    This is a small convenience wrapper around
    :func:`numpy.savez_compressed` that ensures the destination directory
    exists.

    Parameters
    ----------
    X, y:
        Arrays to be stored.
    path:
        Destination file path.  The ``.npz`` extension will be appended if
        missing.

    Returns
    -------
    :class:`pathlib.Path`
        The path to the created archive.
    """

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return path
