from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MacroConfig:
    """Configuration for macro-window hypothesis fitting."""

    k_candidates: tuple[float, ...]
    phase_step: int = 1
    envelope_mode: str = "rms"
    envelope_window: int = 5
    pre_span: int = 3
    post_span: int = 6


@dataclass(frozen=True)
class MacroFitResult:
    k: float
    phase: int
    score: float


@dataclass(frozen=True)
class FirstPeakConfig:
    k: float
    left_lookback: int = 6
    right_lookahead: int = 8
    periodicity_tolerance: float = 0.25


@dataclass(frozen=True)
class FirstPeakSelection:
    indices: tuple[int, ...]
    periodicity_error: float


def _moving_reduce(x: np.ndarray, window: int, fn) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    y = np.asarray(x, dtype=float).reshape(-1)
    if y.size == 0:
        return y
    if window == 1:
        return fn(y[:, None], axis=1)

    out = np.empty_like(y)
    half = window // 2
    for i in range(y.size):
        lo = max(0, i - half)
        hi = min(y.size, lo + window)
        lo = max(0, hi - window)
        out[i] = fn(y[lo:hi])
    return out


def build_envelope(signal: np.ndarray, *, mode: str = "rms", window: int = 5, eps: float = 1e-9) -> np.ndarray:
    x = np.abs(np.asarray(signal, dtype=float).reshape(-1))
    if mode == "rms":
        return np.sqrt(_moving_reduce(x * x, window, np.mean))
    if mode == "max":
        return _moving_reduce(x, window, np.max)
    if mode == "log_energy":
        return np.log(_moving_reduce(x * x, window, np.mean) + eps)
    raise ValueError(f"unknown envelope mode: {mode}")


def flat_to_burst_score(envelope: np.ndarray, idx: int, *, pre_span: int = 3, post_span: int = 6, eps: float = 1e-9) -> float:
    env = np.asarray(envelope, dtype=float).reshape(-1)
    if env.size == 0:
        return float("-inf")
    i = int(np.clip(idx, 0, env.size - 1))
    lo = max(0, i - pre_span)
    pre = env[lo:i] if i > lo else env[max(0, i - 1) : i + 1]
    hi = min(env.size, i + post_span + 1)
    post = env[i:hi]
    pre_level = float(np.mean(pre)) if pre.size else float(env[i])
    post_level = float(np.mean(post)) if post.size else float(env[i])
    return (post_level - pre_level) / (abs(pre_level) + eps)


def fit_macro_k_phase(envelope: np.ndarray, config: MacroConfig) -> MacroFitResult:
    env = np.asarray(envelope, dtype=float).reshape(-1)
    best = MacroFitResult(k=float(config.k_candidates[0]), phase=0, score=float("-inf"))
    for k in config.k_candidates:
        step = max(1, int(round(k)))
        for phase in range(0, step, max(1, config.phase_step)):
            idxs = np.arange(phase, env.size, step)
            if idxs.size == 0:
                continue
            scores = [
                flat_to_burst_score(
                    env,
                    int(i),
                    pre_span=config.pre_span,
                    post_span=config.post_span,
                )
                for i in idxs
            ]
            score = float(np.median(scores))
            if score > best.score:
                best = MacroFitResult(k=float(k), phase=int(phase), score=score)
    return best


def generate_first_peak_candidates(window: np.ndarray, cfg: FirstPeakConfig) -> tuple[int, ...]:
    x = np.asarray(window, dtype=float).reshape(-1)
    if x.size == 0:
        return tuple()
    env = build_envelope(x, mode="max", window=max(3, cfg.left_lookback // 2 + 1))
    start = min(x.size - 1, cfg.left_lookback)
    stop = max(start + 1, min(x.size, int(round(cfg.k)) + cfg.right_lookahead))
    region = env[start:stop]
    if region.size == 0:
        return tuple()
    order = np.argsort(region)[::-1]
    cands = tuple(int(start + i) for i in order[: min(5, order.size)])
    return cands


def select_periodic_first_peak_sequence(candidates_per_window: Iterable[tuple[int, ...]], *, expected_k: float, tolerance: float = 0.25) -> FirstPeakSelection:
    cands = list(candidates_per_window)
    if not cands:
        return FirstPeakSelection(indices=tuple(), periodicity_error=float("inf"))

    chosen: list[int] = []
    for i, options in enumerate(cands):
        if not options:
            continue
        if not chosen:
            chosen.append(int(options[0]))
            continue
        target = chosen[-1] + expected_k
        pick = min(options, key=lambda x: abs(x - target))
        chosen.append(int(pick))

    if len(chosen) < 2:
        return FirstPeakSelection(indices=tuple(chosen), periodicity_error=float("inf"))

    diffs = np.diff(chosen)
    err = float(np.median(np.abs(diffs - expected_k)) / max(expected_k, 1e-9))
    if err > tolerance:
        return FirstPeakSelection(indices=tuple(), periodicity_error=err)
    return FirstPeakSelection(indices=tuple(chosen), periodicity_error=err)


__all__ = [
    "MacroConfig",
    "MacroFitResult",
    "FirstPeakConfig",
    "FirstPeakSelection",
    "build_envelope",
    "flat_to_burst_score",
    "fit_macro_k_phase",
    "generate_first_peak_candidates",
    "select_periodic_first_peak_sequence",
]
