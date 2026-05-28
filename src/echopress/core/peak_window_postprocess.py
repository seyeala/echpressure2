from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from echopress.core.config_io import merge_config, write_resolved_config
from echopress.ingest import load_ostream


def _write_waveform_products_registry(out_dir: Path, summary: dict[str, Any]) -> None:
    products: dict[str, dict[str, Any]] = {}
    candidates = {
        "processed_continuous_train": {"product_name":"processed_continuous_train","kind":"processed","path":"secondary_peak_global_periodic_continuous_train_processed_waveforms.npy","manifest":"global_periodic_continuous_train_manifest.csv","summary":"secondary_peak_processed_summary.json","window_mode":"global-periodic-common","window_output_layout":"continuous-train","horizontal_normalized":True,"vertical_normalized":True,"secondary_peak_suppressed":True,"gain_normalized":True},
        "raw_continuous_train": {"product_name":"raw_continuous_train","kind":"raw","path":"raw_global_periodic_continuous_train_waveforms.npy","manifest":"global_periodic_continuous_train_manifest.csv","summary":"secondary_peak_processed_summary.json","window_mode":"global-periodic-common","window_output_layout":"continuous-train","horizontal_normalized":True,"vertical_normalized":False,"secondary_peak_suppressed":False,"gain_normalized":False},
        "processed_period_rows": {"product_name":"processed_period_rows","kind":"processed","path":"secondary_peak_global_periodic_processed_waveforms.npy","manifest":"global_periodic_window_manifest.csv","summary":"secondary_peak_processed_summary.json","window_mode":"global-periodic-common","window_output_layout":"period-rows"},
        "raw_period_rows": {"product_name":"raw_period_rows","kind":"raw","path":"raw_global_periodic_aligned_waveforms.npy","manifest":"global_periodic_window_manifest.csv","summary":"secondary_peak_processed_summary.json","window_mode":"global-periodic-common","window_output_layout":"period-rows"},
        "canonical_processed": {"product_name":"canonical_processed","kind":"processed","path":"secondary_peak_processed_waveforms.npy","manifest":"secondary_peak_processed_manifest.csv","summary":"secondary_peak_processed_summary.json"},
    }
    for name, meta in candidates.items():
        if (out_dir / meta["path"]).exists() and (out_dir / meta["manifest"]).exists() and (out_dir / meta["summary"]).exists():
            row = dict(meta)
            if name in {"processed_continuous_train", "raw_continuous_train"}:
                row["shape"] = list(summary.get("waveform_shape") or [])
                row["dtype"] = "float32"
            products[name] = row
    default_fft_product = "canonical_processed"
    if "processed_continuous_train" in products:
        default_fft_product = "processed_continuous_train"
    elif "processed_period_rows" in products:
        default_fft_product = "processed_period_rows"
    registry = {"schema_version":"1.0","postprocess_dir":str(out_dir),"default_fft_product":default_fft_product,"products":products}
    (out_dir / "waveform_products.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class PeakWindowPostprocessConfig:
    macro_dir: Path
    echo_dir: Path
    output_dir: Path
    config: Optional[Path] = None
    channel: int = 0
    zero_first_pulse_us: float = 2.0
    peak_neighbor_us: float = 0.0
    gain_clip_min: float = 0.25
    gain_clip_max: float = 4.0
    use_last_common_windows: bool = True
    window_mode: str = "peak-to-peak"
    window_anchor: str = "first"
    window_output_layout: str = "period-rows"
    use_registered_first_peaks: bool = False
    require_registered_first_peaks: bool = False
    periodicity_tolerance_frac: float = 0.12
    max_common_windows: Optional[int] = None
    window_length_samples: Optional[int] = None
    plan_only: bool = False


DEFAULTS: dict[str, Any] = {
    "channel": 0,
    "zero_first_pulse_us": 2.0,
    "peak_neighbor_us": 0.0,
    "gain_clip_min": 0.25,
    "gain_clip_max": 4.0,
    "use_last_common_windows": True,
    "window_mode": "peak-to-peak",
    "window_anchor": "first",
    "window_output_layout": "period-rows",
    "use_registered_first_peaks": False,
    "require_registered_first_peaks": False,
    "periodicity_tolerance_frac": 0.12,
    "max_common_windows": None,
    "window_length_samples": None,
    "plan_only": False,
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


def _load_channel(path: Path, channel: int) -> tuple[np.ndarray, float]:
    o = load_ostream(path, window_mode=False)
    arr = np.asarray(o.channels)
    if arr.ndim == 2:
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
    return y.astype(float), fs


def build_global_periodic_window_plan(first_df: pd.DataFrame, waveform_lengths: dict[str, int], t_global_samples: int, periodicity_tolerance_frac: float, anchor: str = "first", max_common_windows: Optional[int] = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    tol = float(periodicity_tolerance_frac) * float(t_global_samples)
    accepted_by_path: dict[str, list[dict[str, Any]]] = {}
    for path, grp in first_df.groupby("path"):
        n_samples = int(waveform_lengths[str(path)])
        grp = grp.sort_values("first_peak_idx").reset_index(drop=True)
        peaks = grp["first_peak_idx"].astype(int).to_numpy()
        eligible = peaks[peaks + t_global_samples <= n_samples]
        if eligible.size == 0:
            accepted_by_path[str(path)] = []
            continue
        anchor_peak = int(eligible[0] if anchor == "first" else eligible[-1])
        direction = 1 if anchor == "first" else -1
        expected = anchor_peak
        selected = 0
        rows = []
        while True:
            idx = int(np.argmin(np.abs(peaks - expected)))
            nearest = int(peaks[idx])
            err = int(nearest - expected)
            if abs(err) > tol or nearest + t_global_samples > n_samples:
                break
            row = grp.iloc[idx]
            rows.append({"path": str(path), "file": row.get("file", Path(path).name), "pressure_value": row.get("pressure_value", np.nan), "file_index": int(row.get("file_index", -1)) if pd.notna(row.get("file_index", np.nan)) else -1, "selected_window_index": selected, "start_first_peak_idx": nearest, "expected_start_idx": int(expected), "snap_error_samples": err, "snap_error_frac_of_T": float(abs(err) / max(t_global_samples, 1)), "end_idx_exclusive": int(nearest + t_global_samples), "window_len_samples": int(t_global_samples), "T_global_samples": int(t_global_samples), "n_samples": n_samples, "anchor": anchor, "method": "global_periodic_snap"})
            selected += 1
            expected += direction * t_global_samples
        accepted_by_path[str(path)] = sorted(rows, key=lambda x: x["start_first_peak_idx"])
    common_window_count = min((len(v) for v in accepted_by_path.values()), default=0)
    if max_common_windows is not None:
        common_window_count = min(common_window_count, int(max_common_windows))
    if common_window_count <= 0:
        raise RuntimeError("No common fixed-length global-periodic windows found across files.")
    plan = []
    for rows in accepted_by_path.values():
        pick = rows[:common_window_count] if anchor == "first" else rows[-common_window_count:]
        pick = sorted(pick, key=lambda x: x["start_first_peak_idx"])
        for i, r in enumerate(pick):
            rr = dict(r)
            rr["selected_window_index"] = i
            rr["common_window_count"] = common_window_count
            plan.append(rr)
    plan_df = pd.DataFrame(plan).sort_values(["path", "selected_window_index"]).reset_index(drop=True)
    summary = {"window_mode": "global-periodic-common", "window_anchor": anchor, "T_global_samples": int(t_global_samples), "window_samples": int(t_global_samples), "common_window_count": int(common_window_count), "n_files": int(plan_df["path"].nunique()), "n_windows": int(len(plan_df)), "periodicity_tolerance_frac": float(periodicity_tolerance_frac), "max_common_windows": max_common_windows, "method": "global_periodic_snap"}
    return plan_df, summary


def run_peak_window_postprocess(cfg: PeakWindowPostprocessConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    out_dir = Path(rcfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out_dir / "postprocess-peak-windows_config.resolved.yml")
    macro_dir = Path(rcfg["macro_dir"]); echo_dir = Path(rcfg["echo_dir"])
    first_peak_path = macro_dir / "first_peak_index.csv"
    reg_peak_path = macro_dir / "first_peak_index.registered.csv"
    global_window_path = macro_dir / "global_window_size.json"
    echo_peak_path = echo_dir / "echo_peak_index.csv"
    for p in (first_peak_path, global_window_path, echo_peak_path):
        if not p.exists(): raise FileNotFoundError(f"Missing required input: {p}")
    used_peak_table = first_peak_path
    if bool(rcfg["use_registered_first_peaks"]):
        if reg_peak_path.exists(): used_peak_table = reg_peak_path
        elif bool(rcfg["require_registered_first_peaks"]): raise FileNotFoundError(f"Missing required input: {reg_peak_path}")
    first_df = pd.read_csv(used_peak_table).dropna(subset=["path", "first_peak_idx"]).copy(); first_df["first_peak_idx"] = first_df["first_peak_idx"].astype(int)
    echo_df = pd.read_csv(echo_peak_path)
    t_global = int(round(float(json.loads(global_window_path.read_text(encoding="utf-8")).get("T_global_samples", 0))))
    window_len = int(rcfg["window_length_samples"]) if rcfg.get("window_length_samples") is not None else t_global
    if window_len <= 0: raise ValueError("window_length_samples (or T_global_samples) must be > 0")

    if rcfg["window_mode"] == "peak-to-peak":
        rows=[]; segments=[]; max_len=0; skipped=0
        for path, grp in first_df.groupby("path"):
            grp=grp.sort_values("first_peak_idx").reset_index(drop=True)
            if len(grp)<2: continue
            y, fs = _load_channel(Path(path), int(rcfg["channel"]))
            z = int(round(float(rcfg["zero_first_pulse_us"])*1e-6*fs)); nbh = int(round(float(rcfg["peak_neighbor_us"])*1e-6*fs))
            for i in range(len(grp)-1):
                s=int(grp.iloc[i]["first_peak_idx"]); e=int(grp.iloc[i+1]["first_peak_idx"])
                if e<=s or e>len(y): skipped+=1; continue
                raw=y[s:e].astype(float).copy(); proc=raw.copy()
                if z>0: proc[:min(z,len(proc))]=0.0
                if nbh>0:
                    ep=echo_df[(echo_df["path"]==path)&(echo_df["first_peak_idx"]==s)]
                    if len(ep):
                        rel=int(ep["echo_peak_offset_from_first_peak"].min()); lo=max(0,rel-nbh); hi=min(len(proc),rel+nbh+1); proc[lo:hi]=0.0
                gain=float(np.max(np.abs(raw))/max(np.max(np.abs(proc)),1e-12)) if np.max(np.abs(proc))>0 else 1.0
                gain=float(np.clip(gain,float(rcfg["gain_clip_min"]),float(rcfg["gain_clip_max"]))); proc=proc*gain
                segments.append(proc); max_len=max(max_len,len(proc)); rows.append({"path":path,"window_index":i,"start_first_peak_idx":s,"end_first_peak_idx":e,"n_samples":len(raw),"gain":gain})
        if not rows: raise RuntimeError("No complete first_peak[j] -> first_peak[j+1] windows available.")
        raw_aligned=np.zeros((len(rows),max_len),dtype=np.float32); proc_aligned=np.zeros((len(rows),max_len),dtype=np.float32)
        marker_rows=[]; gain_rows=[]
        for i,r in enumerate(rows):
            s=r["n_samples"]; raw=_load_channel(Path(r["path"]), int(rcfg["channel"]))[0][r["start_first_peak_idx"]:r["end_first_peak_idx"]]
            raw_aligned[i,:s]=raw; proc_aligned[i,:s]=segments[i]
            marker_rows.append({"row":i,"start_idx":0,"end_idx_exclusive":s,"t_global_samples":t_global}); gain_rows.append({"row":i,"path":r["path"],"gain":r["gain"]})
        manifest = pd.DataFrame(rows)
        manifest.to_csv(out_dir/"secondary_peak_processed_manifest.csv",index=False)
        pd.DataFrame(gain_rows).to_csv(out_dir/"secondary_peak_gain_table.csv",index=False)
        pd.DataFrame(marker_rows).to_csv(out_dir/"plot_marker_table.csv",index=False)
        np.save(out_dir/"raw_first_peak_to_first_peak_aligned_waveforms.npy",raw_aligned)
        np.save(out_dir/"secondary_peak_processed_waveforms.npy",proc_aligned)
        summary={"window_mode":"peak-to-peak","window_anchor":"last" if rcfg["use_last_common_windows"] else "first","T_global_samples":t_global,"window_samples":None,"common_window_count":None,"n_files":int(manifest["path"].nunique()),"n_windows":int(len(manifest)),"waveform_shape":list(proc_aligned.shape),"periodicity_tolerance_frac":float(rcfg["periodicity_tolerance_frac"]),"max_common_windows":rcfg.get("max_common_windows"),"use_registered_first_peaks":bool(rcfg["use_registered_first_peaks"]),"used_peak_table":str(used_peak_table),"skipped_incomplete":int(skipped),"skipped_bad_periodicity":0,"plan_only":False,"method":"peak_to_peak_pad"}
        (out_dir/"secondary_peak_processed_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        _write_waveform_products_registry(out_dir, summary)
        return summary

    waveforms={}; fs_by_path={}; lengths={}
    for path in first_df["path"].unique():
        y,fs=_load_channel(Path(path), int(rcfg["channel"])); waveforms[str(path)]=y; fs_by_path[str(path)]=fs; lengths[str(path)]=len(y)
    plan_df, plan_summary = build_global_periodic_window_plan(first_df, lengths, window_len, float(rcfg["periodicity_tolerance_frac"]), anchor=str(rcfg["window_anchor"]), max_common_windows=rcfg.get("max_common_windows"))
    plan_df.to_csv(out_dir/"global_periodic_window_plan.csv", index=False)
    summary = dict(plan_summary)
    summary.update({"use_registered_first_peaks":bool(rcfg["use_registered_first_peaks"]),"used_peak_table":str(used_peak_table),"skipped_incomplete":0,"skipped_bad_periodicity":0,"plan_only":bool(rcfg["plan_only"]),"window_output_layout":str(rcfg["window_output_layout"])})
    if rcfg["plan_only"]:
        summary["waveform_shape"]=None
        (out_dir/"secondary_peak_processed_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        _write_waveform_products_registry(out_dir, summary)
        return summary

    layout = str(rcfg["window_output_layout"])
    manifest_rows=[]; gain_rows=[]; marker_rows=[]
    if layout == "continuous-train":
        common_window_count = int(plan_df["common_window_count"].iloc[0])
        train_samples = int(common_window_count * window_len)
        by_path = plan_df.sort_values(["path", "selected_window_index"]).groupby("path", sort=False)
        raw_aligned=np.zeros((len(by_path),train_samples),dtype=np.float32); proc_aligned=np.zeros((len(by_path),train_samples),dtype=np.float32)
        for i, (path, grp) in enumerate(by_path):
            y=waveforms[str(path)]; fs=fs_by_path[str(path)]; z=int(round(float(rcfg["zero_first_pulse_us"])*1e-6*fs)); nbh=int(round(float(rcfg["peak_neighbor_us"])*1e-6*fs))
            first_row = grp.iloc[0]
            train_start = int(first_row["start_first_peak_idx"]); train_end = int(train_start + train_samples)
            raw = y[train_start:train_end]
            if len(raw) != train_samples: raise RuntimeError(f"Incomplete continuous train for {path}: {train_start}:{train_end}")
            proc = raw.copy()
            for _, prow in grp.iterrows():
                k = int(prow["selected_window_index"]); period_offset = k * window_len
                proc[period_offset: min(period_offset + z, len(proc))] = 0.0
                if nbh > 0:
                    ps = int(prow["start_first_peak_idx"])
                    match=echo_df[(echo_df["path"]==path)&(echo_df["first_peak_idx"]==ps)]
                    if len(match):
                        rel=int(match["echo_peak_offset_from_first_peak"].iloc[0]); eo = period_offset + rel
                        lo=max(0,eo-nbh); hi=min(len(proc),eo+nbh+1); proc[lo:hi]=0.0
            gain=float(np.max(np.abs(raw))/max(np.max(np.abs(proc)),1e-12)) if np.max(np.abs(proc))>0 else 1.0
            gain=float(np.clip(gain,float(rcfg["gain_clip_min"]),float(rcfg["gain_clip_max"]))); proc*=gain
            raw_aligned[i,:]=raw; proc_aligned[i,:]=proc
            gain_rows.append({"row":i,"path":path,"gain":gain}); marker_rows.append({"row":i,"start_idx":0,"end_idx_exclusive":train_samples,"t_global_samples":t_global})
            manifest_rows.append({"path":path,"file":first_row.get("file",Path(path).name),"pressure_value":first_row.get("pressure_value",np.nan),"file_index":first_row.get("file_index",-1),"window_index":i,"selected_window_index":0,"start_first_peak_idx":train_start,"end_idx_exclusive":train_end,"n_samples":train_samples,"T_global_samples":int(first_row["T_global_samples"]),"window_mode":"global-periodic-common","window_anchor":rcfg["window_anchor"],"window_output_layout":"continuous-train","common_window_count":common_window_count,"gain":gain})
        manifest=pd.DataFrame(manifest_rows)
        manifest.to_csv(out_dir/"global_periodic_continuous_train_manifest.csv",index=False)
        plan_df.to_csv(out_dir/"global_periodic_continuous_train_plan.csv",index=False)
        np.save(out_dir/"raw_global_periodic_continuous_train_waveforms.npy",raw_aligned)
        np.save(out_dir/"secondary_peak_global_periodic_continuous_train_processed_waveforms.npy",proc_aligned)
    else:
        raw_aligned=np.zeros((len(plan_df),window_len),dtype=np.float32); proc_aligned=np.zeros((len(plan_df),window_len),dtype=np.float32)
        for i,row in plan_df.iterrows():
            path=str(row["path"]); y=waveforms[path]; fs=fs_by_path[path]; s=int(row["start_first_peak_idx"]); e=s+window_len; raw=y[s:e]
            if len(raw)!=window_len: raise RuntimeError(f"Incomplete fixed window for {path}: {s}:{e}")
            proc=raw.copy(); z=int(round(float(rcfg["zero_first_pulse_us"])*1e-6*fs)); nbh=int(round(float(rcfg["peak_neighbor_us"])*1e-6*fs))
            proc[:min(z,len(proc))]=0.0
            if nbh>0:
                match=echo_df[(echo_df["path"]==path)&(echo_df["first_peak_idx"]==s)]
                if len(match):
                    rel=int(match["echo_peak_offset_from_first_peak"].iloc[0]); lo=max(0,rel-nbh); hi=min(len(proc),rel+nbh+1); proc[lo:hi]=0.0
            gain=float(np.max(np.abs(raw))/max(np.max(np.abs(proc)),1e-12)) if np.max(np.abs(proc))>0 else 1.0
            gain=float(np.clip(gain,float(rcfg["gain_clip_min"]),float(rcfg["gain_clip_max"]))); proc*=gain
            raw_aligned[i,:]=raw; proc_aligned[i,:]=proc
            gain_rows.append({"row":i,"path":path,"gain":gain}); marker_rows.append({"row":i,"start_idx":0,"end_idx_exclusive":window_len,"t_global_samples":t_global})
            manifest_rows.append({"path":path,"file":row.get("file",Path(path).name),"pressure_value":row.get("pressure_value",np.nan),"file_index":row.get("file_index",-1),"window_index":i,"selected_window_index":int(row["selected_window_index"]),"start_first_peak_idx":s,"end_idx_exclusive":e,"n_samples":int(row["n_samples"]),"T_global_samples":int(row["T_global_samples"]),"window_mode":"global-periodic-common","window_anchor":rcfg["window_anchor"],"window_output_layout":"period-rows","common_window_count":int(row["common_window_count"]),"snap_error_samples":int(row["snap_error_samples"]),"snap_error_frac_of_T":float(row["snap_error_frac_of_T"]),"gain":gain})
        manifest=pd.DataFrame(manifest_rows)
        manifest.to_csv(out_dir/"global_periodic_window_manifest.csv",index=False)
    manifest.to_csv(out_dir/"secondary_peak_processed_manifest.csv",index=False)
    pd.DataFrame(gain_rows).to_csv(out_dir/"secondary_peak_gain_table.csv",index=False)
    pd.DataFrame(marker_rows).to_csv(out_dir/"plot_marker_table.csv",index=False)
    np.save(out_dir/"raw_global_periodic_aligned_waveforms.npy",raw_aligned)
    np.save(out_dir/"secondary_peak_global_periodic_processed_waveforms.npy",proc_aligned)
    np.save(out_dir/"raw_first_peak_to_first_peak_aligned_waveforms.npy",raw_aligned)
    np.save(out_dir/"secondary_peak_processed_waveforms.npy",proc_aligned)
    train_samples = int(plan_df["common_window_count"].iloc[0]) * int(window_len)
    summary.update({"T_global_samples":t_global,"window_samples":window_len,"common_window_count":int(plan_df["common_window_count"].iloc[0]),"n_files":int(manifest["path"].nunique()),"n_windows":int(len(manifest)),"waveform_shape":list(proc_aligned.shape),"period_samples":int(window_len),"train_samples":train_samples,"method":"global_periodic_common_fixed"})
    (out_dir/"secondary_peak_processed_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    _write_waveform_products_registry(out_dir, summary)
    return summary
