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

    in_csv = Path(rcfg["postprocess_dir"]) / "postprocessed_peak_windows.csv"
    if not in_csv.exists():
        raise FileNotFoundError("fft-postprocessed requires postprocessed_peak_windows.csv from postprocess-peak-windows")
    df = pd.read_csv(in_csv)
    bins = int(rcfg["fft_bins"])
    offsets = pd.to_numeric(df.get("first_echo_offset", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy(dtype=float)
    signal = offsets - np.mean(offsets) if offsets.size else np.zeros(1)
    fft = np.abs(np.fft.rfft(signal, n=bins))
    np.save(out_dir / "postprocessed_fft.npy", fft.astype(np.float32))
    pd.DataFrame({"bin": np.arange(len(fft)), "magnitude": fft}).to_csv(out_dir / "postprocessed_fft.csv", index=False)
    summary = {"n_rows": int(len(df)), "fft_bins": bins, "n_fft_points": int(len(fft))}
    (out_dir / "fft_postprocessed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
