from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd

@dataclass(frozen=True)
class PressureEvalConfig:
    dataset_dir: Path
    model_dir: Path
    split: str = "test"

def run_evaluate(cfg: PressureEvalConfig):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    ds=Path(cfg.dataset_dir); md=Path(cfg.model_dir); out=md/f"eval_{cfg.split}"; out.mkdir(parents=True, exist_ok=True)
    X=np.load(ds/"X.npy"); y=np.load(ds/"y.npy"); split=json.loads((ds/"split.json").read_text())
    idx=np.array(split[cfg.split]); prep=json.loads((md/"preprocessing.json").read_text())
    Xs=(X[idx]-np.array(prep["x_mean"])) / np.array(prep["x_std"])
    model=tf.keras.models.load_model(md/"model_best.keras")
    pred=(model.predict(Xs, verbose=0).reshape(-1)*prep["y_std"]+prep["y_mean"])
    yt=y[idx]; resid=pred-yt
    mae=float(np.mean(np.abs(resid))); rmse=float(np.sqrt(np.mean(resid**2))); bias=float(np.mean(resid)); med=float(np.median(np.abs(resid))); mmax=float(np.max(np.abs(resid))); r2=float(1.0 - np.sum(resid**2)/max(np.sum((yt-np.mean(yt))**2),1e-12))
    (out/"metrics_test.json").write_text(json.dumps({"mae":mae,"rmse":rmse,"r2":r2,"bias":bias,"median_abs_error":med,"max_abs_error":mmax,"n_test":int(len(idx))}, indent=2), encoding="utf-8")
    pd.DataFrame({"y_true":yt,"y_pred":pred,"residual":resid}).to_csv(out/"predictions_test.csv", index=False)
    plt.figure(); plt.scatter(yt,pred,s=10); plt.xlabel("true"); plt.ylabel("pred"); plt.tight_layout(); plt.savefig(out/"prediction_vs_true.png", dpi=170); plt.close()
