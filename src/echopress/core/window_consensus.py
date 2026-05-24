from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KAggregate:
    k: float
    median: float
    trimmed_mean: float
    count: int


def _trimmed_mean(values: np.ndarray, trim: float) -> float:
    vals = np.sort(np.asarray(values, dtype=float))
    if vals.size == 0:
        return float("nan")
    ntrim = int(np.floor(vals.size * trim))
    if 2 * ntrim >= vals.size:
        return float(np.mean(vals))
    return float(np.mean(vals[ntrim : vals.size - ntrim]))


def aggregate_per_k(scores_by_k: dict[float, list[float]], *, trim_fraction: float = 0.1) -> dict[float, KAggregate]:
    out: dict[float, KAggregate] = {}
    for k, scores in scores_by_k.items():
        vals = np.asarray(scores, dtype=float)
        if vals.size == 0:
            continue
        out[float(k)] = KAggregate(
            k=float(k),
            median=float(np.median(vals)),
            trimmed_mean=_trimmed_mean(vals, trim_fraction),
            count=int(vals.size),
        )
    return out


def select_global_k(aggregates: dict[float, KAggregate]) -> float:
    if not aggregates:
        raise ValueError("no aggregates")
    ranked = sorted(aggregates.values(), key=lambda a: (a.trimmed_mean, a.median), reverse=True)
    return float(ranked[0].k)


def refit_with_global_k(per_file_candidates: dict[str, dict[float, float]], global_k: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, mapping in per_file_candidates.items():
        if global_k in mapping:
            out[name] = float(mapping[global_k])
        elif mapping:
            nearest = min(mapping, key=lambda k: abs(k - global_k))
            out[name] = float(mapping[nearest])
    return out


def mad_outlier_flags(values: np.ndarray, *, z_thresh: float = 3.5) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        return np.zeros_like(x, dtype=bool)
    mz = 0.6745 * (x - med) / mad
    return np.abs(mz) > z_thresh


__all__ = [
    "KAggregate",
    "aggregate_per_k",
    "select_global_k",
    "refit_with_global_k",
    "mad_outlier_flags",
]
