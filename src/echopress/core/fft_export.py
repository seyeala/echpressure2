from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from echopress.core.config_io import merge_config, write_resolved_config


@dataclass(frozen=True)
class FFTExportConfig:
    postprocess_dir: Path
    output_dir: Path
    config: Optional[Path] = None
    fft_bins: Optional[int] = 256


def _resolve_config(cfg: FFTExportConfig) -> dict[str, Any]:
    default_yml = Path(__file__).resolve().parents[3] / "configs" / "fft_export.default.yml"
    rcfg = merge_config(default_yaml_path=default_yml, user_yaml_path=cfg.config, cli_values=asdict(cfg))
    rcfg["postprocess_dir"] = str(cfg.postprocess_dir)
    rcfg["output_dir"] = str(cfg.output_dir)
    return rcfg


def run_fft_postprocessed(cfg: FFTExportConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    out_dir = Path(rcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out_dir / "fft-postprocessed_config.resolved.yml")

    post_dir = Path(rcfg["postprocess_dir"])
    in_waveforms = post_dir / "secondary_peak_processed_waveforms.npy"
    in_manifest = post_dir / "secondary_peak_processed_manifest.csv"
    in_summary = post_dir / "secondary_peak_processed_summary.json"
    missing = [str(p) for p in [in_waveforms, in_manifest, in_summary] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "fft-postprocessed requires outputs from postprocess-peak-windows:\n" + "\n".join(missing)
        )

    waveforms = np.load(in_waveforms)
    manifest = pd.read_csv(in_manifest)
    _ = json.loads(in_summary.read_text(encoding="utf-8"))

    if waveforms.ndim != 2:
        raise ValueError(f"secondary_peak_processed_waveforms.npy must be 2D [n_files, n_samples], got shape={waveforms.shape}")
    if len(manifest) != int(waveforms.shape[0]):
        raise ValueError(
            f"row count mismatch: secondary_peak_processed_manifest.csv has {len(manifest)} rows but waveforms has {waveforms.shape[0]}"
        )

    bins = int(rcfg["fft_bins"])
    n_samples = int(waveforms.shape[1])
    fft_complex = np.fft.rfft(waveforms, n=bins, axis=1)
    fft_mag = np.abs(fft_complex).astype(np.float32)
    fft_db = (20.0 * np.log10(np.maximum(fft_mag, 1e-12))).astype(np.float32)
    row_max = np.max(fft_db, axis=1, keepdims=True)
    fft_relative_db = (fft_db - row_max).astype(np.float32)
    fft_cycles_per_window = np.fft.rfftfreq(bins, d=1.0).astype(np.float32) * float(n_samples)

    np.save(out_dir / "fft_mag.npy", fft_mag)
    np.save(out_dir / "fft_db.npy", fft_db)
    np.save(out_dir / "fft_relative_db.npy", fft_relative_db)
    np.save(out_dir / "fft_cycles_per_window.npy", fft_cycles_per_window)
    manifest.to_csv(out_dir / "fft_manifest.csv", index=False)

    summary = {
        "n_rows": int(waveforms.shape[0]),
        "window_samples": n_samples,
        "fft_bins": bins,
        "n_fft_points": int(fft_mag.shape[1]),
    }
    (out_dir / "fft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
