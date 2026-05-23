from __future__ import annotations

"""Template-Constrained Incident Marker Localization (TCIML).

Algorithm 2 implementation for phase-constrained marker localization using
raw/envelope normalized cross-correlation.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks, peak_prominences


@dataclass(frozen=True)
class TCIMLConfig:
    T_hat: float
    T_error_samples: float
    peak_width_samples: int
    search_radius_min: int = 8
    alpha: float = 0.75
    C_min: float = 0.25
    P_min: float = 0.0
    W_minus: int = 8
    W_plus: int = 8
    envelope_rel_threshold: float = 0.35
    template_peak_prominence: float = 0.0


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


def _extract_envelope(file_obj: Any, adapters: Any, signal: np.ndarray) -> np.ndarray:
    if isinstance(adapters, dict) and "load_envelope" in adapters:
        env = np.asarray(adapters["load_envelope"](file_obj), dtype=float)
        return np.ravel(env)
    if isinstance(adapters, dict) and "hte" in adapters and callable(adapters["hte"]):
        env = np.asarray(adapters["hte"](signal), dtype=float)
        return np.ravel(env)
    return np.abs(signal)


def _norm(a: np.ndarray) -> np.ndarray:
    x = a.astype(float)
    x = x - float(np.mean(x))
    s = float(np.linalg.norm(x))
    if s <= 1e-12:
        return np.zeros_like(x)
    return x / s


def _estimate_phase(peaks: np.ndarray, T_hat: float) -> float:
    if peaks.size == 0:
        return 0.0
    mods = np.mod(peaks, T_hat)
    return float(np.median(mods))


def _expected_centers(N: int, phase: float, cfg: TCIMLConfig) -> np.ndarray:
    T = cfg.T_hat
    w = cfg.peak_width_samples
    k_min = int(np.floor((0 - phase) / T)) - 1
    k_max = int(np.ceil((N - phase) / T)) + 1
    centers = []
    for k in range(k_min, k_max + 1):
        c = phase + k * T
        lo = int(np.floor(c - w))
        hi = int(np.ceil(c + w + 1))
        if lo >= 0 and hi <= N:
            centers.append(int(round(c)))
    return np.asarray(centers, dtype=int)


def _ncc(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    t = _norm(template)
    w = template.size
    if signal.size < w:
        return np.asarray([], dtype=float)
    conv = correlate(signal, t[::-1], mode="valid")
    out = np.zeros_like(conv, dtype=float)
    for i in range(conv.size):
        seg = _norm(signal[i : i + w])
        out[i] = float(np.dot(seg, t))
    return out


def _build_template(signals: list[np.ndarray], cfg: TCIMLConfig) -> tuple[np.ndarray, np.ndarray | None]:
    w = cfg.peak_width_samples
    snippets: list[np.ndarray] = []
    env_snippets: list[np.ndarray] = []

    for sig in signals:
        if sig.size < 2 * w + 1:
            continue
        pks, props = find_peaks(np.abs(sig), prominence=cfg.template_peak_prominence, distance=max(1, int(0.5 * cfg.T_hat)))
        if pks.size == 0:
            continue
        prom = props.get("prominences", np.ones(pks.size))
        keep = pks[prom >= max(cfg.P_min, float(np.median(prom)))]
        for p in keep:
            lo, hi = p - w, p + w + 1
            if lo < 0 or hi > sig.size:
                continue
            snippets.append(_norm(sig[lo:hi]))
            env_snippets.append(_norm(np.abs(sig[lo:hi])))

    if not snippets:
        raise ValueError("no template snippets could be built")

    raw_tpl = np.median(np.asarray(snippets), axis=0)
    env_tpl = np.median(np.asarray(env_snippets), axis=0) if env_snippets else None
    return _norm(raw_tpl), (_norm(env_tpl) if env_tpl is not None else None)


def run_tciml(
    files: Sequence[Any],
    period_summary: dict[str, Any],
    per_file_period_df: pd.DataFrame,
    config: TCIMLConfig,
    adapters: Any = None,
) -> pd.DataFrame:
    del period_summary, per_file_period_df

    rows: list[dict[str, Any]] = []
    file_signals: list[np.ndarray] = []
    file_ids: list[str] = []

    for i, f in enumerate(files):
        file_id = str(getattr(f, "name", f"file_{i}"))
        sig = _extract_signal(f, adapters)
        file_ids.append(file_id)
        file_signals.append(sig)

    raw_template, env_template = _build_template(file_signals, config)
    np.save("incident_template.npy", raw_template)
    if env_template is not None:
        np.save("incident_template_env.npy", env_template)

    w = config.peak_width_samples
    for file_id, sig in zip(file_ids, file_signals):
        N = sig.size
        cand_peaks, _ = find_peaks(np.abs(sig), distance=max(1, int(0.5 * config.T_hat)))
        phase = _estimate_phase(cand_peaks.astype(float), config.T_hat)
        expected = _expected_centers(N, phase, config)
        env = _extract_envelope(file_id, adapters, sig)

        for mu in expected:
            R = int(max(3 * config.T_error_samples, 2 * w, config.search_radius_min))
            lo = max(0, mu - R - w)
            hi = min(N, mu + R + w + 1)
            segment = sig[lo:hi]
            if segment.size < raw_template.size:
                continue
            raw_ncc = _ncc(segment, raw_template)
            if raw_ncc.size == 0:
                continue
            raw_idx = int(np.argmax(raw_ncc))
            raw_score = float(raw_ncc[raw_idx])
            center = lo + raw_idx + w

            env_score = np.nan
            total_score = raw_score
            if env_template is not None and env.size == sig.size:
                env_seg = env[lo:hi]
                env_ncc = _ncc(env_seg, env_template)
                if env_ncc.size > raw_idx:
                    env_score = float(env_ncc[raw_idx])
                    total_score = config.alpha * raw_score + (1.0 - config.alpha) * env_score

            # onset/end by envelope threshold crossing
            onset = center - config.W_minus
            end = center + config.W_plus
            local_env = np.abs(sig[max(0, center - w) : min(N, center + w + 1)])
            if local_env.size > 5:
                th = config.envelope_rel_threshold * float(np.max(local_env))
                left = np.where(np.abs(sig[max(0, center - w) : center + 1]) <= th)[0]
                right = np.where(np.abs(sig[center : min(N, center + w + 1)]) <= th)[0]
                if left.size > 0:
                    onset = max(0, max(0, center - w) + int(left[-1]))
                if right.size > 0:
                    end = min(N - 1, center + int(right[0]))

            residual = abs(center - mu)
            prom = float(peak_prominences(np.abs(sig), np.asarray([center], dtype=int))[0][0]) if 0 < center < N - 1 else 0.0
            amp = float(np.abs(sig[center])) if 0 <= center < N else 0.0

            accepted = True
            reject: list[str] = []
            if total_score < config.C_min:
                accepted = False
                reject.append("score_low")
            if residual > R:
                accepted = False
                reject.append("residual_large")
            if prom < config.P_min:
                accepted = False
                reject.append("prominence_low")

            rows.append(
                {
                    "file_id": file_id,
                    "expected_center_idx": int(mu),
                    "matched_center_idx": int(center),
                    "window_start_idx": int(max(0, onset - config.W_minus)),
                    "window_end_idx": int(min(N - 1, end + config.W_plus)),
                    "onset_idx": int(onset),
                    "end_idx": int(end),
                    "raw_ncc": raw_score,
                    "env_ncc": env_score,
                    "blended_score": total_score,
                    "residual_samples": float(residual),
                    "local_prominence": prom,
                    "local_amplitude": amp,
                    "accepted": accepted,
                    "reject_reason": ";".join(reject) if reject else "",
                }
            )

    marker_df = pd.DataFrame(rows)
    marker_df.to_csv("incident_marker_table.csv", index=False)

    meta = {
        "algorithm": "tciml",
        "config": asdict(config),
        "n_files": len(files),
        "n_markers": int(marker_df.shape[0]),
        "n_accepted": int(marker_df["accepted"].sum()) if not marker_df.empty else 0,
    }
    Path("incident_marker_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return marker_df
