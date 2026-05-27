from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from echopress.core.config_io import merge_config, write_resolved_config
from echopress.ingest import load_ostream


@dataclass(frozen=True)
class PeakWindowPostprocessConfig:
    macro_dir: Path
    echo_dir: Path
    output_dir: Path
    config: Optional[Path] = None
    zero_first_pulse_us: float = 2.0
    peak_neighbor_us: float = 0.0
    gain_clip_min: float = 0.25
    gain_clip_max: float = 4.0
    use_last_common_windows: bool = True


DEFAULTS: dict[str, Any] = {
    "zero_first_pulse_us": 2.0,
    "peak_neighbor_us": 0.0,
    "gain_clip_min": 0.25,
    "gain_clip_max": 4.0,
    "use_last_common_windows": True,
}


def _resolve_config(cfg: PeakWindowPostprocessConfig) -> dict[str, Any]:
    rcfg = dict(DEFAULTS)
    if cfg.config is not None:
        rcfg = merge_config(default_yaml_path=None, user_yaml_path=cfg.config, cli_values=asdict(cfg))
    else:
        rcfg.update(asdict(cfg))
    rcfg["macro_dir"] = str(cfg.macro_dir)
    rcfg["echo_dir"] = str(cfg.echo_dir)
    rcfg["output_dir"] = str(cfg.output_dir)
    return rcfg


def _load_channel(path: Path) -> tuple[np.ndarray, float]:
    o = load_ostream(path, window_mode=False)
    arr = np.asarray(o.channels)
    y = arr[:, 0] if arr.ndim == 2 else arr.reshape(-1)
    ts = np.asarray(getattr(o, "timestamps", []), dtype=float).reshape(-1)
    fs = 1.0
    if ts.size > 3:
        d = np.diff(ts)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            fs = float(1.0 / np.median(d))
    return y.astype(float), fs


def run_peak_window_postprocess(cfg: PeakWindowPostprocessConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    out_dir = Path(rcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out_dir / "postprocess-peak-windows_config.resolved.yml")

    first_peak_path = Path(rcfg["macro_dir"]) / "first_peak_index.csv"
    global_window_path = Path(rcfg["macro_dir"]) / "global_window_size.json"
    echo_peak_path = Path(rcfg["echo_dir"]) / "echo_peak_index.csv"
    for p in (first_peak_path, global_window_path, echo_peak_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    first_df = pd.read_csv(first_peak_path).dropna(subset=["path", "first_peak_idx"]).copy()
    first_df["first_peak_idx"] = first_df["first_peak_idx"].astype(int)
    echo_df = pd.read_csv(echo_peak_path)

    with global_window_path.open("r", encoding="utf-8") as f:
        g = json.load(f)
    t_global = int(round(float(g.get("T_global_samples", 0))))

    rows: list[dict[str, Any]] = []
    segments: list[np.ndarray] = []
    max_len = 0
    skipped = 0
    for path, grp in first_df.groupby("path"):
        grp = grp.sort_values("first_peak_idx").reset_index(drop=True)
        if len(grp) < 2:
            continue
        if bool(rcfg["use_last_common_windows"]):
            pairs = list(range(len(grp) - 1))
        else:
            pairs = list(range(len(grp) - 1))
        y, fs = _load_channel(Path(path))
        z = int(round(float(rcfg["zero_first_pulse_us"]) * 1e-6 * fs))
        nbh = int(round(float(rcfg["peak_neighbor_us"]) * 1e-6 * fs))
        for i in pairs:
            s = int(grp.iloc[i]["first_peak_idx"])
            e = int(grp.iloc[i + 1]["first_peak_idx"])
            if e <= s or e > len(y):
                skipped += 1
                continue
            raw = y[s:e].astype(float).copy()
            proc = raw.copy()
            if z > 0:
                proc[: min(z, len(proc))] = 0.0
            if nbh > 0:
                rel = None
                ep = echo_df[(echo_df["path"] == path) & (echo_df["first_peak_idx"] == s)]
                if len(ep):
                    rel = int(ep["echo_peak_offset_from_first_peak"].min())
                if rel is not None:
                    lo = max(0, rel - nbh)
                    hi = min(len(proc), rel + nbh + 1)
                    proc[lo:hi] = 0.0
            gain = float(np.max(np.abs(raw)) / max(np.max(np.abs(proc)), 1e-12)) if np.max(np.abs(proc)) > 0 else 1.0
            gain = float(np.clip(gain, float(rcfg["gain_clip_min"]), float(rcfg["gain_clip_max"])))
            proc = proc * gain
            segments.append(proc)
            max_len = max(max_len, len(proc))
            rows.append({"path": path, "window_index": i, "start_first_peak_idx": s, "end_first_peak_idx": e, "n_samples": len(raw), "gain": gain})

    if not rows:
        raise RuntimeError("No complete first_peak[j] -> first_peak[j+1] windows available.")

    raw_aligned = np.zeros((len(rows), max_len), dtype=np.float32)
    proc_aligned = np.zeros((len(rows), max_len), dtype=np.float32)
    marker_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        s = r["n_samples"]
        raw = _load_channel(Path(r["path"]))[0][r["start_first_peak_idx"]:r["end_first_peak_idx"]]
        raw_aligned[i, :s] = raw
        proc_aligned[i, :s] = segments[i]
        marker_rows.append({"row": i, "start_idx": 0, "end_idx_exclusive": s, "t_global_samples": t_global})
        gain_rows.append({"row": i, "path": r["path"], "gain": r["gain"]})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "secondary_peak_processed_manifest.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(out_dir / "secondary_peak_gain_table.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(out_dir / "plot_marker_table.csv", index=False)
    np.save(out_dir / "raw_first_peak_to_first_peak_aligned_waveforms.npy", raw_aligned)
    np.save(out_dir / "secondary_peak_processed_waveforms.npy", proc_aligned)

    summary = {"n_files": int(manifest["path"].nunique()), "n_windows": int(len(manifest)), "skipped_incomplete": int(skipped)}
    (out_dir / "secondary_peak_processed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
