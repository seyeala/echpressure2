from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from echopress.core.alignment_edit import load_alignment_rows
from echopress.ingest import load_ostream


def resolve_path(path: str | Path, dataset_root: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p

    root = Path(dataset_root)
    matches = list(root.rglob(p.name))
    if matches:
        return matches[0]

    return p


def load_signal(path: str | Path, channel: int = 0) -> tuple[np.ndarray, float | None]:
    """
    Returns:
      signal: 1D waveform
      fs: estimated sampling rate if timestamps support it, else None
    """
    ostream = load_ostream(path, window_mode=False)
    channels = np.asarray(ostream.channels)

    if channels.ndim == 1:
        signal = channels.astype(float).reshape(-1)
    elif channels.ndim == 2 and channels.shape[1] > channel:
        signal = channels[:, channel].astype(float).reshape(-1)
    else:
        raise ValueError(f"No usable channel {channel} in {path}")

    timestamps = np.asarray(ostream.timestamps, dtype=float)
    fs = None

    if timestamps.size > 2:
        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            fs = 1.0 / float(np.median(diffs))

    return signal, fs


def baseline_sample_count(
    *,
    baseline_samples: int | None,
    baseline_seconds: float | None,
    fs: float | None,
    n_signal: int,
) -> int:
    if baseline_samples is not None:
        n = int(baseline_samples)
    elif baseline_seconds is not None:
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError("baseline_seconds requires valid timestamps/fs")
        n = int(round(float(baseline_seconds) * fs))
    else:
        raise ValueError("Specify either baseline_samples or baseline_seconds")

    if n <= 0:
        raise ValueError("Baseline window must contain at least one sample")

    return min(n, n_signal)


def amplitude_metrics(
    signal: np.ndarray,
    *,
    baseline_samples: int | None = None,
    baseline_seconds: float | None = None,
    fs: float | None = None,
) -> dict[str, Any]:
    y = np.asarray(signal, dtype=float).reshape(-1)

    if y.size == 0:
        raise ValueError("Empty signal")

    n_base = baseline_sample_count(
        baseline_samples=baseline_samples,
        baseline_seconds=baseline_seconds,
        fs=fs,
        n_signal=y.size,
    )

    abs_y = np.abs(y)
    baseline_mean_abs = float(np.mean(abs_y[:n_base]))
    max_abs = float(np.max(abs_y))
    peak_idx = int(np.argmax(abs_y))
    peak_to_baseline_ratio = (
        float(max_abs / baseline_mean_abs) if baseline_mean_abs > 0 else float("inf")
    )

    return {
        "n_samples": int(y.size),
        "baseline_samples_used": int(n_base),
        "baseline_mean_abs": baseline_mean_abs,
        "max_abs": max_abs,
        "peak_idx": peak_idx,
        "peak_to_baseline_ratio": peak_to_baseline_ratio,
    }


def build_low_peak_remove_list(
    *,
    align_table: str | Path,
    dataset_root: str | Path,
    output_list: str | Path,
    channel: int = 0,
    baseline_samples: int | None = None,
    baseline_seconds: float | None = None,
    threshold_multiplier: float = 1.0,
    include_missing: bool = True,
) -> dict[str, Any]:
    """
    Adds a row to the remove list if:

        max(abs(signal)) <= threshold_multiplier * mean(abs(signal[:x]))

    where x is baseline_samples, or baseline_seconds converted to samples.
    """
    rows = load_alignment_rows(align_table)

    remove_items: list[dict[str, Any]] = []
    checked = 0
    kept = 0
    missing = 0
    errors = 0

    for idx, row in enumerate(rows):
        raw_path = row.get("path")
        if not raw_path:
            errors += 1
            continue

        resolved = resolve_path(str(raw_path), dataset_root)

        if not resolved.exists():
            missing += 1
            if include_missing:
                remove_items.append(
                    {
                        "row_index": idx,
                        "path": str(raw_path),
                        "resolved_path": str(resolved),
                        "sid": row.get("sid"),
                        "file_stamp": row.get("file_stamp"),
                        "pressure_value": row.get("pressure_value"),
                        "reason": "missing_file",
                    }
                )
            continue

        try:
            signal, fs = load_signal(resolved, channel=channel)
            metrics = amplitude_metrics(
                signal,
                baseline_samples=baseline_samples,
                baseline_seconds=baseline_seconds,
                fs=fs,
            )
            checked += 1

            threshold = threshold_multiplier * metrics["baseline_mean_abs"]
            should_remove = metrics["max_abs"] <= threshold

            if should_remove:
                remove_items.append(
                    {
                        "row_index": idx,
                        "path": str(raw_path),
                        "resolved_path": str(resolved),
                        "sid": row.get("sid"),
                        "file_stamp": row.get("file_stamp"),
                        "pressure_value": row.get("pressure_value"),
                        "alignment_error": row.get("alignment_error"),
                        "reason": "max_abs_not_bigger_than_baseline_mean_abs",
                        "threshold_multiplier": threshold_multiplier,
                        "threshold": threshold,
                        **metrics,
                    }
                )
            else:
                kept += 1

        except Exception as exc:
            errors += 1
            remove_items.append(
                {
                    "row_index": idx,
                    "path": str(raw_path),
                    "resolved_path": str(resolved),
                    "sid": row.get("sid"),
                    "file_stamp": row.get("file_stamp"),
                    "pressure_value": row.get("pressure_value"),
                    "reason": "load_or_metric_error",
                    "error": str(exc),
                }
            )

    out = Path(output_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(remove_items, indent=2, default=float))

    return {
        "align_table": str(align_table),
        "dataset_root": str(dataset_root),
        "output_list": str(output_list),
        "input_rows": len(rows),
        "checked_rows": checked,
        "kept_rows": kept,
        "missing_files": missing,
        "error_rows": errors,
        "remove_rows": len(remove_items),
        "channel": channel,
        "baseline_samples": baseline_samples,
        "baseline_seconds": baseline_seconds,
        "threshold_multiplier": threshold_multiplier,
    }
