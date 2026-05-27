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
    fft_mode: str = "full"
    n_fft: Optional[int] = None
    output_bins: Optional[int] = None


def _reduce_spectrum_bins(spectrum: np.ndarray, output_bins: int) -> np.ndarray:
    current_bins = int(spectrum.shape[1])
    if output_bins >= current_bins:
        return spectrum
    edges = np.linspace(0, current_bins, output_bins + 1)
    reduced = np.empty((spectrum.shape[0], output_bins), dtype=np.float32)
    for i in range(output_bins):
        lo = int(np.floor(edges[i]))
        hi = int(np.floor(edges[i + 1]))
        if hi <= lo:
            hi = min(lo + 1, current_bins)
        reduced[:, i] = np.mean(spectrum[:, lo:hi], axis=1)
    return reduced


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

    n_samples = int(waveforms.shape[1])
    fft_mode = str(rcfg.get("fft_mode", "full"))
    if fft_mode not in {"full", "truncate", "resample-spectrum"}:
        raise ValueError("fft_mode must be one of: full, truncate, resample-spectrum")
    n_fft = int(rcfg["n_fft"]) if rcfg.get("n_fft") is not None else n_samples
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")

    bins = rcfg.get("output_bins")
    if bins is None:
        bins = rcfg.get("fft_bins")
    output_bins = int(bins) if bins is not None else None
    if output_bins is not None and output_bins <= 0:
        raise ValueError("output_bins/fft_bins must be positive")

    cropped_time_domain = False
    if fft_mode == "truncate":
        legacy_bins = output_bins if output_bins is not None else int(rcfg.get("fft_bins") or 256)
        fft_complex = np.fft.rfft(waveforms, n=legacy_bins, axis=1)
        effective_n_fft = legacy_bins
        cropped_time_domain = legacy_bins < n_samples
    else:
        fft_complex = np.fft.rfft(waveforms, n=n_fft, axis=1)
        effective_n_fft = n_fft

    fft_mag = np.abs(fft_complex).astype(np.float32)
    if output_bins is not None and output_bins < fft_mag.shape[1]:
        fft_mag = _reduce_spectrum_bins(fft_mag, output_bins)
    fft_db = (20.0 * np.log10(np.maximum(fft_mag, 1e-12))).astype(np.float32)
    row_max = np.max(fft_db, axis=1, keepdims=True)
    fft_relative_db = (fft_db - row_max).astype(np.float32)

    full_cycles = np.fft.rfftfreq(effective_n_fft, d=1.0).astype(np.float32) * float(n_samples)
    if output_bins is not None and output_bins < full_cycles.shape[0]:
        target_x = np.linspace(0.0, 1.0, output_bins, dtype=np.float32)
        source_x = np.linspace(0.0, 1.0, full_cycles.shape[0], dtype=np.float32)
        fft_cycles_per_window = np.interp(target_x, source_x, full_cycles).astype(np.float32)
    else:
        fft_cycles_per_window = full_cycles

    np.save(out_dir / "fft_mag.npy", fft_mag)
    np.save(out_dir / "fft_db.npy", fft_db)
    np.save(out_dir / "fft_relative_db.npy", fft_relative_db)
    np.save(out_dir / "fft_cycles_per_window.npy", fft_cycles_per_window)
    manifest.to_csv(out_dir / "fft_manifest.csv", index=False)

    summary = {
        "n_rows": int(waveforms.shape[0]),
        "window_samples": n_samples,
        "waveform_samples": n_samples,
        "fft_bins": output_bins,
        "output_bins": output_bins,
        "fft_mode": fft_mode,
        "n_fft": effective_n_fft,
        "cropped_time_domain": cropped_time_domain,
        "n_fft_points": int(fft_mag.shape[1]),
    }
    (out_dir / "fft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
