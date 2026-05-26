from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from echopress.core.config_io import write_resolved_config

QCStage = Literal["macro", "echo", "postprocess", "fft"]


@dataclass(frozen=True)
class QCPlotConfig:
    stage: QCStage
    input_dir: Path
    output_dir: Path


def run_qc_plot(cfg: QCPlotConfig) -> dict[str, object]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config({"stage": cfg.stage, "input_dir": str(cfg.input_dir), "output_dir": str(cfg.output_dir)}, cfg.output_dir / f"plot-{cfg.stage}-qc_config.resolved.yml")
    fig, ax = plt.subplots(figsize=(8, 4))
    if cfg.stage == "macro":
        df = pd.read_csv(cfg.input_dir / "first_peak_index.csv")
        ax.hist(pd.to_numeric(df["first_peak_idx"], errors="coerce").dropna(), bins=50)
    elif cfg.stage == "echo":
        df = pd.read_csv(cfg.input_dir / "echo_peak_index.csv")
        ax.hist(pd.to_numeric(df["echo_peak_offset_from_first_peak"], errors="coerce").dropna(), bins=50)
    elif cfg.stage == "postprocess":
        df = pd.read_csv(cfg.input_dir / "postprocessed_peak_windows.csv")
        ax.hist(pd.to_numeric(df["n_echo_peaks_post"], errors="coerce").dropna(), bins=20)
    else:
        arr = np.load(cfg.input_dir / "fft_relative_db.npy")
        arr = np.mean(arr, axis=0) if arr.ndim == 2 else arr
        ax.plot(arr)
    ax.set_title(f"{cfg.stage} qc")
    out = cfg.output_dir / f"plot-{cfg.stage}-qc.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    summary = {"stage": cfg.stage, "plot": str(out)}
    (cfg.output_dir / f"plot_{cfg.stage}_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
