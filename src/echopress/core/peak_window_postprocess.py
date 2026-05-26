from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from echopress.core.config_io import merge_config, write_resolved_config


@dataclass(frozen=True)
class PeakWindowPostprocessConfig:
    echo_dir: Path
    output_dir: Path
    config: Optional[Path] = None
    max_echo_peak_order: Optional[int] = 3


def _resolve_config(cfg: PeakWindowPostprocessConfig) -> dict[str, Any]:
    default_yml = Path(__file__).resolve().parents[3] / "configs" / "peak_window_postprocess.default.yml"
    rcfg = merge_config(default_yaml_path=default_yml, user_yaml_path=cfg.config, cli_values=asdict(cfg))
    rcfg["echo_dir"] = str(cfg.echo_dir)
    rcfg["output_dir"] = str(cfg.output_dir)
    return rcfg


def run_peak_window_postprocess(cfg: PeakWindowPostprocessConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    out_dir = Path(rcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out_dir / "postprocess-peak-windows_config.resolved.yml")

    echo_path = Path(rcfg["echo_dir"]) / "echo_peak_index.csv"
    windows_path = Path(rcfg["echo_dir"]) / "echo_window_index.csv"
    waveforms_path = Path(rcfg["echo_dir"]) / "echo_window_values.npy"
    if not echo_path.exists() or not windows_path.exists() or not waveforms_path.exists():
        raise FileNotFoundError(
            "postprocess-peak-windows requires echo_peak_index.csv, echo_window_index.csv, and echo_window_values.npy from detect-echo-peaks"
        )

    echo = pd.read_csv(echo_path)
    windows = pd.read_csv(windows_path)
    waveforms = np.load(waveforms_path)

    if rcfg.get("max_echo_peak_order") is not None and "echo_peak_order" in echo.columns:
        echo = echo[echo["echo_peak_order"] <= int(rcfg["max_echo_peak_order"])]

    features = (
        echo.groupby(["path", "first_peak_idx"], dropna=False)
        .agg(n_echo_peaks_post=("echo_peak_idx", "count"), first_echo_offset=("echo_peak_offset_from_first_peak", "min"))
        .reset_index()
    )
    merged = windows.merge(features, how="left", on=["path", "first_peak_idx"])
    merged["n_echo_peaks_post"] = merged["n_echo_peaks_post"].fillna(0).astype(int)

    if waveforms.ndim != 2:
        raise ValueError(f"echo_window_values.npy must be 2D [n_files, window_samples], got shape={waveforms.shape}")
    if len(merged) != int(waveforms.shape[0]):
        raise ValueError(
            f"row count mismatch: echo_window_index.csv has {len(merged)} rows but echo_window_values.npy has {waveforms.shape[0]} windows"
        )

    np.save(out_dir / "secondary_peak_processed_waveforms.npy", waveforms.astype(np.float32))
    merged.to_csv(out_dir / "secondary_peak_processed_manifest.csv", index=False)

    summary = {
        "n_windows": int(len(merged)),
        "n_echo_peaks": int(len(echo)),
        "waveform_length": int(waveforms.shape[1]),
    }
    (out_dir / "secondary_peak_processed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
