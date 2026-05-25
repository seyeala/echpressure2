from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert

from echopress.core.config_io import merge_config, write_resolved_config
from echopress.ingest import load_ostream


@dataclass(frozen=True)
class EchoPeakConfig:
    detection_dir: Path
    output_dir: Optional[Path] = None
    config: Optional[Path] = None

    channel: Optional[int] = None
    use_registered: Optional[bool] = None

    zero_before_us: Optional[float] = None
    zero_after_us: Optional[float] = None
    zero_before_samples: Optional[int] = None
    zero_after_samples: Optional[int] = None

    hilbert_frac: Optional[float] = None
    min_prominence_rel: Optional[float] = None
    min_height_rel: Optional[float] = None
    min_distance_samples: Optional[int] = None
    refine_radius_samples: Optional[int] = None
    max_peaks_per_window: Optional[int] = None
    fallback_to_t_global_window_end: Optional[bool] = None

    save_cleaned_windows: Optional[bool] = None
    progress_every: Optional[int] = None
    quiet: Optional[bool] = None


DEFAULTS: dict[str, Any] = {
    "channel": 0,
    "use_registered": False,
    "zero_before_us": 0.0,
    "zero_after_us": 2.0,
    "zero_before_samples": 0,
    "zero_after_samples": 2000,
    "hilbert_frac": 0.20,
    "min_prominence_rel": 0.08,
    "min_height_rel": 0.05,
    "min_distance_samples": 200,
    "refine_radius_samples": 80,
    "max_peaks_per_window": 8,
    "fallback_to_t_global_window_end": False,
    "save_cleaned_windows": False,
    "progress_every": 25,
    "quiet": False,
}



def _resolve_config(cfg: EchoPeakConfig) -> dict[str, Any]:
    resolved = dict(DEFAULTS)
    if cfg.config is not None:
        resolved = merge_config(
            default_yaml_path=Path("configs/echo_peaks.default.yml"),
            user_yaml_path=cfg.config,
            cli_values=asdict(cfg),
        )
    else:
        resolved.update({k: v for k, v in asdict(cfg).items() if v is not None})
    resolved["detection_dir"] = str(cfg.detection_dir)
    resolved["output_dir"] = str(cfg.output_dir or (cfg.detection_dir / "echo_peaks"))
    return resolved
def _load_channel(path: Path, channel: int) -> tuple[np.ndarray, float]:
    o = load_ostream(path, window_mode=False)
    arr = np.asarray(o.channels)
    if arr.ndim == 2:
        if arr.shape[1] <= channel:
            raise ValueError(f"channel {channel} unavailable for {path}; shape={arr.shape}")
        y = arr[:, channel]
    else:
        y = arr.reshape(-1)
    ts = np.asarray(getattr(o, "timestamps", []), dtype=float).reshape(-1)
    fs = 1.0
    if ts.size > 3:
        d = np.diff(ts)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            fs = float(1.0 / np.median(d))
    return np.nan_to_num(y.astype(float), nan=0.0, posinf=0.0, neginf=0.0), fs

def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y"})

def _progress(stage: str, i: int, total: int, t0: float) -> str:
    elapsed = time.time() - t0
    rate = i / elapsed if elapsed > 0 else 0.0
    eta = (total - i) / rate if rate > 0 else float("nan")
    return f"[{stage}] {i}/{total} files | elapsed={elapsed/60:.1f} min | rate={rate:.2f} files/s | ETA={eta/60:.1f} min"

def _get_T_global(detection_dir: Path, peaks: pd.DataFrame) -> float:
    summary_path = detection_dir / "global_window_size.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        value = summary.get("T_global_samples")
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value)
    diffs: list[float] = []
    for _, grp in peaks.groupby("path"):
        pk = np.sort(grp["first_peak_idx"].astype(int).to_numpy())
        if len(pk) > 1:
            diffs.extend(np.diff(pk).astype(float).tolist())
    if not diffs:
        raise RuntimeError("Could not determine T_global_samples from global_window_size.json or peak spacings.")
    return float(np.median(np.asarray(diffs, dtype=float)))

def run_echo_peak_detection(cfg: EchoPeakConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    detection_dir = Path(rcfg["detection_dir"])
    output_dir = Path(rcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    use_registered = bool(rcfg["use_registered"])
    registered_path = detection_dir / "first_peak_index.registered.csv"
    all_first_path = detection_dir / "first_peak_index.csv"
    peak_path = registered_path if use_registered else all_first_path
    if not peak_path.exists():
        raise FileNotFoundError(f"missing required peak table: {peak_path}")
    peaks = pd.read_csv(peak_path)
    required = {"path", "first_peak_idx", "pressure_value"}
    missing = required - set(peaks.columns)
    if missing:
        raise ValueError(f"{peak_path} missing columns {missing}; columns={list(peaks.columns)}")
    if use_registered and "used_for_backward_common_window" in peaks.columns:
        mask = _bool_series(peaks["used_for_backward_common_window"])
        if mask.any():
            peaks = peaks[mask].copy()
    peaks = peaks.dropna(subset=["path", "first_peak_idx"]).copy()
    peaks["first_peak_idx"] = peaks["first_peak_idx"].astype(int)
    if peaks.empty:
        raise RuntimeError(f"No usable first peak rows in {peak_path}")
    T_global = _get_T_global(detection_dir, peaks)
    T_int = int(round(T_global))
    channel = int(rcfg["channel"])
    hilbert_frac = float(rcfg["hilbert_frac"])
    min_prominence_rel = float(rcfg["min_prominence_rel"])
    min_height_rel = float(rcfg["min_height_rel"])
    min_distance_samples = int(rcfg["min_distance_samples"])
    refine_radius_samples = int(rcfg["refine_radius_samples"])
    max_peaks_per_window = int(rcfg["max_peaks_per_window"])
    fallback_to_t_global_window_end = bool(rcfg["fallback_to_t_global_window_end"])
    save_cleaned_windows = bool(rcfg["save_cleaned_windows"])
    progress_every = int(rcfg["progress_every"])
    quiet = bool(rcfg["quiet"])
    if not (0 < hilbert_frac <= 1.0):
        raise ValueError("--hilbert-frac must be in (0, 1].")
    if max_peaks_per_window <= 0:
        raise ValueError("--max-peaks-per-window must be positive.")
    write_resolved_config(rcfg, output_dir / "detect-echo-peaks_config.resolved.yml")
    file_groups = list(peaks.groupby("path"))
    total_files = len(file_groups)
    echo_rows=[]; window_rows=[]; cleaned_windows=[]; cleaned_index=[]
    t0 = time.time()
    for file_i,(path_str,grp) in enumerate(file_groups,start=1):
        if not quiet and (file_i == 1 or file_i % progress_every == 0 or file_i == total_files):
            print(_progress("echo-peak pass", file_i, total_files, t0), flush=True)
        y, fs = _load_channel(Path(path_str), channel)
        if fs > 1:
            zero_after = int(round(float(rcfg["zero_after_us"]) * 1e-6 * fs))
            zero_before = int(round(float(rcfg["zero_before_us"]) * 1e-6 * fs))
        else:
            zero_before = int(rcfg["zero_before_samples"])
            zero_after = int(rcfg["zero_after_samples"])
        zero_before=max(0,zero_before); zero_after=max(1,zero_after)
        grp = grp.sort_values(["first_peak_idx"]).reset_index(drop=True)
        first_peaks_sorted = grp["first_peak_idx"].astype(int).to_numpy()
        for local_i,r in grp.iterrows():
            fp=int(r["first_peak_idx"])
            win_start=fp
            next_first_peak_idx = int(first_peaks_sorted[local_i + 1]) if local_i + 1 < len(first_peaks_sorted) else None
            if next_first_peak_idx is not None:
                win_end = min(len(y), next_first_peak_idx)
                window_end_source = "next_first_peak"
            elif fallback_to_t_global_window_end:
                win_end = min(len(y), fp + T_int)
                window_end_source = "t_global_fallback"
            else:
                win_end = len(y)
                window_end_source = "signal_end"
            if win_end <= win_start + 8: continue
            raw=y[win_start:win_end].astype(float); clean=raw-float(np.median(raw))
            zero_end_local=min(len(clean),zero_after); clean[0:zero_end_local]=0.0
            search_start_local=zero_end_local; search_end_local=min(len(clean), int(round(hilbert_frac*T_int)))
            window_base={"path":path_str,"file":Path(path_str).name,"pressure_value":float(r["pressure_value"]),"file_index":int(r["file_index"]) if "file_index" in r and pd.notna(r["file_index"]) else file_i-1,"macro_window_index":int(r["macro_window_index"]) if "macro_window_index" in r and pd.notna(r["macro_window_index"]) else local_i,"registered_window_index":local_i,"T_global_samples":float(T_global),"first_peak_idx":fp,"window_start_idx":win_start,"window_end_idx_exclusive":win_end,"window_end_source":window_end_source,"next_first_peak_idx":next_first_peak_idx if next_first_peak_idx is not None else pd.NA,"zero_start_idx":fp,"zero_end_idx_exclusive":min(len(y), fp+zero_after),"zero_before_samples":zero_before,"zero_after_samples":zero_after,"hilbert_search_start_idx":fp+search_start_local,"hilbert_search_end_idx_exclusive":fp+search_end_local}
            if search_end_local <= search_start_local + 4:
                window_rows.append({**window_base,"n_echo_peaks":0,"status":"search_region_empty"}); continue
            env=np.abs(hilbert(clean)); env_search=env[search_start_local:search_end_local]; env_max=float(np.nanmax(env_search)) if env_search.size else 0.0
            if not np.isfinite(env_max) or env_max <= 0:
                window_rows.append({**window_base,"n_echo_peaks":0,"status":"zero_envelope"}); continue
            peaks_local,props=find_peaks(env_search,height=min_height_rel*env_max,prominence=min_prominence_rel*env_max,distance=max(1,min_distance_samples))
            if len(peaks_local):
                prominences=props.get("prominences", np.zeros(len(peaks_local))); heights=props.get("peak_heights", env_search[peaks_local]); order=np.argsort(prominences)[::-1][:max_peaks_per_window]
                peaks_local=peaks_local[order]; prominences=prominences[order]; heights=heights[order]; tord=np.argsort(peaks_local); peaks_local=peaks_local[tord]; prominences=prominences[tord]; heights=heights[tord]
            else:
                prominences=np.asarray([]); heights=np.asarray([])
            window_rows.append({**window_base,"n_echo_peaks":int(len(peaks_local)),"status":"OK"})
            for echo_order,(pk_local,pk_height,pk_prom) in enumerate(zip(peaks_local,heights,prominences),start=1):
                env_peak_idx=int(fp+search_start_local+pk_local); refine_lo=max(win_start,env_peak_idx-refine_radius_samples); refine_hi=min(win_end,env_peak_idx+refine_radius_samples+1)
                refined=refine_lo+int(np.argmax(np.abs(y[refine_lo:refine_hi]))) if refine_hi>refine_lo else env_peak_idx
                echo_rows.append({**window_base,"echo_peak_order":int(echo_order),"echo_peak_idx":int(refined),"echo_peak_value":float(y[refined]),"echo_peak_abs_value":float(abs(y[refined])),"echo_peak_offset_from_first_peak":int(refined-fp),"echo_peak_offset_frac_of_T":float((refined-fp)/max(T_global,1.0)),"hilbert_envelope_peak_idx":int(env_peak_idx),"hilbert_envelope_value":float(env[env_peak_idx-fp]) if 0<=env_peak_idx-fp<len(env) else float(pk_height),"hilbert_envelope_height":float(pk_height),"hilbert_envelope_prominence":float(pk_prom),"method":"hilbert_envelope_then_raw_abs_refine"})
            if save_cleaned_windows:
                cleaned_windows.append(clean.astype(np.float32)); cleaned_index.append({**window_base,"cleaned_window_row":len(cleaned_windows)-1})
    echo_df=pd.DataFrame(echo_rows); window_df=pd.DataFrame(window_rows)
    echo_df.to_csv(output_dir/"echo_peak_index.csv",index=False); window_df.to_csv(output_dir/"echo_window_index.csv",index=False)
    if save_cleaned_windows and cleaned_windows:
        max_len=max(len(x) for x in cleaned_windows); arr=np.zeros((len(cleaned_windows),max_len),dtype=np.float32); lengths=[]
        for i,x in enumerate(cleaned_windows): arr[i,:len(x)]=x; lengths.append(len(x))
        np.save(output_dir/"cleaned_windows.npy",arr); ci=pd.DataFrame(cleaned_index); ci["cleaned_window_valid_samples"]=lengths; ci.to_csv(output_dir/"cleaned_window_index.csv",index=False)
    summary={"detection_dir":str(detection_dir),"output_dir":str(output_dir),"used_peak_table":str(peak_path),"T_global_samples":float(T_global),"n_files":int(total_files),"n_windows":int(len(window_df)),"n_echo_peaks":int(len(echo_df)),"method":"hilbert_envelope_then_raw_abs_refine"}
    (output_dir/"echo_peak_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary
