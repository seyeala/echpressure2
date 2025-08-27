"""Dataset construction utilities.

The functions defined here offer a very small abstraction for building in-memory
``(X, y)`` datasets from various sources.  They are intentionally lightweight so
that they can be re-used in notebooks or scripts without pulling in heavy
framework dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np

from .to_numpy import dataframe_to_numpy

DataSource = Union[pd.DataFrame, Path, str]


def load_dataframe(source: DataSource, *, read_csv_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Load a :class:`~pandas.DataFrame` from *source*.

    Parameters
    ----------
    source:
        Either a dataframe itself or a filesystem path pointing to a CSV file.
    read_csv_kwargs:
        Optional keyword arguments forwarded to :func:`pandas.read_csv` when
        ``source`` is a path.
    """

    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)
    read_csv_kwargs = read_csv_kwargs or {}
    return pd.read_csv(path, **read_csv_kwargs)


def build_dataset(
    source: DataSource,
    target: str,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    save_csv: Optional[Path] = None,
    save_npz: Optional[Path] = None,
    read_csv_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create ``(X, y)`` arrays from *source*.

    This convenience function combines :func:`load_dataframe` and
    :func:`export.to_numpy.dataframe_to_numpy`.
    """

    df = load_dataframe(source, read_csv_kwargs=read_csv_kwargs)
    return dataframe_to_numpy(
        df,
        target=target,
        feature_columns=feature_columns,
        save_csv=save_csv,
        save_npz=save_npz,
    )
