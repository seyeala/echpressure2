from __future__ import annotations
import numpy as np, pandas as pd

def make_pressure_splits(y, manifest, cfg):
    train_frac = float(cfg.get("train_frac",0.70)); val_frac=float(cfg.get("val_frac",0.15));
    n = len(y); rng = np.random.default_rng(int(cfg.get("random_seed",42)))
    idx = np.arange(n)
    method = cfg.get("method","pressure_stratified")
    train=[]; val=[]; test=[]
    if method == "pressure_stratified":
        q = int(cfg.get("pressure_bins",10))
        bins = pd.qcut(pd.Series(y), q=min(q, len(np.unique(y))), duplicates="drop")
        for _, g in pd.Series(idx).groupby(bins):
            a = g.to_numpy(); rng.shuffle(a); ntr=int(round(len(a)*train_frac)); nv=int(round(len(a)*val_frac))
            train.extend(a[:ntr]); val.extend(a[ntr:ntr+nv]); test.extend(a[ntr+nv:])
    else:
        a = idx.copy(); rng.shuffle(a); ntr=int(round(n*train_frac)); nv=int(round(n*val_frac))
        train=a[:ntr].tolist(); val=a[ntr:ntr+nv].tolist(); test=a[ntr+nv:].tolist()
    return {"train": sorted(map(int,train)), "val": sorted(map(int,val)), "test": sorted(map(int,test)), "method": method, "random_seed": int(cfg.get("random_seed",42))}
