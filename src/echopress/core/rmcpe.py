from __future__ import annotations

"""Robust Multi-file Comb Period Estimation (RMCPE).

Algorithm 1 implementation for estimating a global window period from
multiple signals.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RMCPEConfig:
    T_min: float
    T_max: float
    raw_max_abs_min: float = 0.0
    max_env_points: int = 55000
    prominence: float = 0.0
    distance: int = 1
    width: float | None = None
    tau_T: float = 1.0
    lambda_: float = 1.0
    robust_loss: str = "huber"
    bootstrap_count: int = 200
    random_seed: int = 0
    poor_comb_score_min: float = 1e-6


@dataclass(frozen=True)
class FileFitResult:
    file_id: str
    accepted: bool
    reject_reason: str | None
    T_i: float | None
    phase_i: float | None
    score: float
    residual_median: float | None
    residual_mad: float | None
    n_peaks: int


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _block_max_envelope(signal: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = signal.size
    B_i = max(1, int(np.ceil(n / max_points)))
    n_blocks = int(np.ceil(n / B_i))
    env = np.zeros(n_blocks, dtype=float)
    idx_map = np.zeros(n_blocks, dtype=int)
    abs_signal = np.abs(signal)
    for j in range(n_blocks):
        lo = j * B_i
        hi = min((j + 1) * B_i, n)
        block = abs_signal[lo:hi]
        if block.size == 0:
            continue
        local = int(np.argmax(block))
        env[j] = float(block[local])
        idx_map[j] = lo + local
    return env, idx_map, B_i


def _robust_weighted_score(residuals: np.ndarray, cfg: RMCPEConfig) -> float:
    scale = max(_mad(residuals), 1e-9)
    x = residuals / scale
    if cfg.robust_loss == "huber":
        a = np.where(np.abs(x) <= cfg.tau_T, 0.5 * x * x, cfg.tau_T * (np.abs(x) - 0.5 * cfg.tau_T))
    elif cfg.robust_loss == "cauchy":
        a = np.log1p((x / cfg.tau_T) ** 2)
    else:
        a = np.abs(x)
    return float(np.exp(-cfg.lambda_ * np.mean(a)))


def _phase_circular_median(peaks: np.ndarray, T: float) -> float:
    mods = np.mod(peaks, T)
    candidates = np.linspace(0.0, T, 360, endpoint=False)
    dist = np.min(np.stack([np.abs(mods[:, None] - candidates[None, :]), T - np.abs(mods[:, None] - candidates[None, :])]), axis=0)
    return float(candidates[np.argmin(np.sum(dist, axis=0))])


def _extract_signal(file_obj: Any, adapters: Any) -> np.ndarray:
    if adapters is None:
        arr = np.asarray(file_obj, dtype=float)
    elif callable(adapters):
        arr = np.asarray(adapters(file_obj), dtype=float)
    elif isinstance(adapters, dict) and "load_signal" in adapters:
        arr = np.asarray(adapters["load_signal"](file_obj), dtype=float)
    else:
        raise TypeError("adapters must be None, callable, or dict with 'load_signal'")
    return np.ravel(arr)


def _fit_file(file_id: str, signal: np.ndarray, cfg: RMCPEConfig) -> FileFitResult:
    if signal.size == 0:
        return FileFitResult(file_id, False, "empty", None, None, 0.0, None, None, 0)
    if not np.isfinite(signal).all():
        return FileFitResult(file_id, False, "nan_inf", None, None, 0.0, None, None, 0)
    if float(np.max(np.abs(signal))) < cfg.raw_max_abs_min:
        return FileFitResult(file_id, False, "weak_signal", None, None, 0.0, None, None, 0)

    env, idx_map, B_i = _block_max_envelope(signal, cfg.max_env_points)
    min_dist_env = max(1, int(np.ceil(cfg.T_min / B_i)))
    peaks_env, _ = find_peaks(env, prominence=cfg.prominence, distance=max(cfg.distance, min_dist_env), width=cfg.width)
    peaks = idx_map[peaks_env]
    if peaks.size < 3:
        return FileFitResult(file_id, False, "insufficient_peaks", None, None, 0.0, None, None, int(peaks.size))

    cands: list[float] = []
    for l in range(1, 5):
        if peaks.size <= l:
            continue
        d = (peaks[l:] - peaks[:-l]) / float(l)
        keep = d[(d >= cfg.T_min) & (d <= cfg.T_max)]
        cands.extend(keep.tolist())
    if not cands:
        return FileFitResult(file_id, False, "insufficient_peaks", None, None, 0.0, None, None, int(peaks.size))

    cand_arr = np.asarray(cands, dtype=float)
    T_i = float(np.median(cand_arr))
    phase_i = _phase_circular_median(peaks.astype(float), T_i)
    k = np.rint((peaks - phase_i) / T_i)
    tooth = phase_i + k * T_i
    residuals = peaks - tooth
    score = _robust_weighted_score(residuals, cfg)
    reason = None if score >= cfg.poor_comb_score_min else "poor_comb_score"
    return FileFitResult(
        file_id=file_id,
        accepted=reason is None,
        reject_reason=reason,
        T_i=T_i,
        phase_i=phase_i,
        score=score,
        residual_median=float(np.median(residuals)),
        residual_mad=_mad(residuals),
        n_peaks=int(peaks.size),
    )


def run_rmcpe(files: Sequence[Any], config: RMCPEConfig, adapters: Any = None) -> tuple[dict[str, Any], pd.DataFrame]:
    results: list[FileFitResult] = []
    for i, f in enumerate(files):
        file_id = str(getattr(f, "name", f"file_{i}"))
        try:
            signal = _extract_signal(f, adapters)
            res = _fit_file(file_id, signal, config)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("rmcpe failed for %s", file_id)
            res = FileFitResult(file_id, False, f"error:{type(exc).__name__}", None, None, 0.0, None, None, 0)
        if not res.accepted:
            LOGGER.warning("rmcpe rejected file %s reason=%s", file_id, res.reject_reason)
        results.append(res)

    df = pd.DataFrame([asdict(r) for r in results])
    accepted = df[df["accepted"]]

    if accepted.empty:
        T_hat = float("nan")
        T_error = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        T_vals = accepted["T_i"].to_numpy(dtype=float)
        local_scores = accepted["score"].to_numpy(dtype=float)
        g_lo = max(config.T_min, float(np.min(T_vals) - 2.0 * np.median(np.abs(T_vals - np.median(T_vals)))))
        g_hi = min(config.T_max, float(np.max(T_vals) + 2.0 * np.median(np.abs(T_vals - np.median(T_vals)))))
        grid = np.linspace(g_lo, g_hi, 400)
        sigma = max(_mad(T_vals), 1e-6)
        total = np.array([np.sum(local_scores * np.exp(-0.5 * ((T_vals - g) / sigma) ** 2)) for g in grid])
        T_hat = float(grid[int(np.argmax(total))])
        T_error = 1.4826 * _mad(T_vals)

        rng = np.random.default_rng(config.random_seed)
        boots = []
        for _ in range(config.bootstrap_count):
            sample = rng.choice(T_vals, size=T_vals.size, replace=True)
            boots.append(float(np.median(sample)))
        ci_low, ci_high = [float(x) for x in np.percentile(boots, [2.5, 97.5])]

    summary = {
        "algorithm": "rmcpe",
        "config": asdict(config),
        "n_files": len(files),
        "n_accepted": int(accepted.shape[0]),
        "T_hat": T_hat,
        "T_error_samples": T_error,
        "T_bootstrap_ci": [ci_low, ci_high],
    }

    Path("window_period_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv("window_period_per_file.csv", index=False)
    return summary, df
