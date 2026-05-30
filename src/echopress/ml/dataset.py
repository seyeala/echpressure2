from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from echopress.core.config_io import apply_dotted_overrides, merge_config, write_resolved_config
from .splits import make_pressure_splits

@dataclass(frozen=True)
class PressureDatasetConfig:
    fft_dir: Path
    output_dir: Path
    config: Optional[Path] = None
    feature_source: Optional[str] = None
    target_column: Optional[str] = None
    freq_min_cycles_per_window: Optional[float] = None
    freq_max_cycles_per_window: Optional[float] = None
    bin_average: Optional[int] = None
    require_finite: Optional[bool] = None
    drop_nan_targets: Optional[bool] = None
    overrides: Optional[List[str]] = None

def _resolve_config(cfg: PressureDatasetConfig) -> dict[str, Any]:
    default_yml = Path(__file__).resolve().parents[3] / "configs" / "pressure_regression.default.yml"
    rcfg = merge_config(default_yaml_path=default_yml, user_yaml_path=cfg.config, cli_values=asdict(cfg))
    rcfg = apply_dotted_overrides(rcfg, cfg.overrides)
    for k in ("feature_source","target_column","freq_min_cycles_per_window","freq_max_cycles_per_window","bin_average","require_finite","drop_nan_targets"):
        if k in rcfg.get("dataset", {}) and rcfg.get(k) is None:
            rcfg[k] = rcfg["dataset"][k]
    rcfg.pop("overrides", None)
    rcfg["fft_dir"] = str(cfg.fft_dir); rcfg["output_dir"] = str(cfg.output_dir)
    if cfg.config is not None:
        rcfg["config"] = str(cfg.config)
    else:
        rcfg.pop("config", None)
    return rcfg

def build_pressure_dataset(cfg: PressureDatasetConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    out = Path(rcfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out / "dataset_config.resolved.yml")
    fft_dir = Path(rcfg["fft_dir"])
    manifest = pd.read_csv(fft_dir / "fft_manifest.csv")
    target = rcfg.get("target_column", "pressure_value")
    y = pd.to_numeric(manifest[target], errors="coerce").to_numpy(dtype=float)
    if rcfg.get("drop_nan_targets", True):
        keep = np.isfinite(y)
        manifest = manifest.loc[keep].reset_index(drop=True); y = y[keep]
    src = str(rcfg.get("feature_source", "fft_relative_db"))
    name = {"fft_relative_db":"fft_relative_db.npy","fft-relative-db":"fft_relative_db.npy","fft_db":"fft_db.npy","fft-db":"fft_db.npy","fft_mag":"fft_mag.npy","fft-mag":"fft_mag.npy"}.get(src, "fft_relative_db.npy")
    x_all = np.load(fft_dir / name)
    if len(x_all) != len(pd.read_csv(fft_dir / "fft_manifest.csv")):
        raise ValueError("feature rows do not match fft_manifest.csv")
    if rcfg.get("drop_nan_targets", True): x_all = x_all[keep]
    freq = np.load(fft_dir / "fft_cycles_per_window.npy")
    m = (freq >= float(rcfg.get("freq_min_cycles_per_window",0.0))) & (freq <= float(rcfg.get("freq_max_cycles_per_window", np.max(freq))))
    X = x_all[:, m]; f = freq[m]
    b = int(rcfg.get("bin_average",1) or 1)
    if b > 1:
        n = (X.shape[1] // b) * b
        X = X[:, :n].reshape(X.shape[0], -1, b).mean(axis=2)
        f = f[:n].reshape(-1,b).mean(axis=1)
    if rcfg.get("require_finite", True):
        good = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[good]; y = y[good]; manifest = manifest.loc[good].reset_index(drop=True)
    splits = make_pressure_splits(y, manifest, rcfg.get("split", {}))
    split_map = np.array(["train"] * len(y), dtype=object)
    split_map[splits["val"]] = "val"; split_map[splits["test"]] = "test"
    manifest = manifest.copy(); manifest["row"] = np.arange(len(manifest)); manifest["split"] = split_map
    cols = [c for c in ["row","path","file","pressure_value","split"] if c in manifest.columns]
    manifest[cols].to_csv(out / "sample_manifest.csv", index=False)
    np.save(out / "X.npy", X.astype(np.float32)); np.save(out / "y.npy", y.astype(np.float32)); np.save(out / "feature_axis.npy", f.astype(np.float32))
    (out / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = {"n_samples": int(len(y)), "n_features": int(X.shape[1]), "feature_source": src}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
