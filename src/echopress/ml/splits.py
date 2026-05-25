from __future__ import annotations
import numpy as np, pandas as pd

def _random_split(indices, train_frac, val_frac, rng):
    a = np.array(indices, dtype=int)
    rng.shuffle(a)
    ntr = int(round(len(a) * train_frac))
    nv = int(round(len(a) * val_frac))
    return a[:ntr], a[ntr:ntr+nv], a[ntr+nv:]

def make_pressure_splits(y, manifest, cfg):
    train_frac = float(cfg.get("train_frac", 0.70)); val_frac = float(cfg.get("val_frac", 0.15))
    rng = np.random.default_rng(int(cfg.get("random_seed", 42)))
    idx = np.arange(len(y), dtype=int)
    method = cfg.get("method", "pressure_stratified")
    train, val, test = [], [], []
    if method == "pressure_stratified":
        q = int(cfg.get("pressure_bins", 10))
        bins = pd.qcut(pd.Series(y), q=min(q, len(np.unique(y))), duplicates="drop")
        for _, g in pd.Series(idx).groupby(bins):
            tr, va, te = _random_split(g.to_numpy(), train_frac, val_frac, rng)
            train.extend(tr.tolist()); val.extend(va.tolist()); test.extend(te.tolist())
    elif method == "by_file_stamp" and "file_stamp" in manifest.columns:
        groups = manifest.groupby("file_stamp").indices.values()
        group_ids = np.arange(len(groups))
        tr_g, va_g, te_g = _random_split(group_ids, train_frac, val_frac, rng)
        for i in tr_g: train.extend(np.asarray(list(groups[i]), dtype=int).tolist())
        for i in va_g: val.extend(np.asarray(list(groups[i]), dtype=int).tolist())
        for i in te_g: test.extend(np.asarray(list(groups[i]), dtype=int).tolist())
    elif method == "holdout_pressure_range":
        pmin = cfg.get("holdout_pressure_min"); pmax = cfg.get("holdout_pressure_max")
        if pmin is None or pmax is None:
            raise ValueError("holdout_pressure_range requires holdout_pressure_min and holdout_pressure_max")
        holdout = (y >= float(pmin)) & (y <= float(pmax))
        test = idx[holdout].tolist()
        tr, va, _ = _random_split(idx[~holdout], train_frac / max(train_frac + val_frac, 1e-12), 1.0 - (train_frac / max(train_frac + val_frac, 1e-12)), rng)
        train.extend(tr.tolist()); val.extend(va.tolist())
    else:
        tr, va, te = _random_split(idx, train_frac, val_frac, rng)
        train, val, test = tr.tolist(), va.tolist(), te.tolist()
    return {"train": sorted(map(int, train)), "val": sorted(map(int, val)), "test": sorted(map(int, test)), "method": method, "random_seed": int(cfg.get("random_seed", 42))}
