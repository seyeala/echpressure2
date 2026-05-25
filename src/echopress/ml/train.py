from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd
from echopress.core.config_io import merge_config, write_resolved_config
from .models import build_model
from .preprocess import fit_transform_preprocess

@dataclass(frozen=True)
class PressureTrainConfig:
    dataset_dir: Path
    output_dir: Path
    config: Optional[Path] = None

def run_train(cfg: PressureTrainConfig):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install with: pip install -r requirements-ml.txt") from exc
    default_yml = Path(__file__).resolve().parents[3] / "configs" / "pressure_regression.default.yml"
    rcfg = merge_config(default_yaml_path=default_yml, user_yaml_path=cfg.config, cli_values={"dataset_dir":str(cfg.dataset_dir),"output_dir":str(cfg.output_dir)})
    out=Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True); write_resolved_config(rcfg, out/"train_config.resolved.yml")
    X=np.load(Path(cfg.dataset_dir)/"X.npy"); y=np.load(Path(cfg.dataset_dir)/"y.npy"); split=json.loads((Path(cfg.dataset_dir)/"split.json").read_text())
    tr,va,te=np.array(split["train"]),np.array(split["val"]),np.array(split["test"])
    Xtr,Xva,Xte, ytr,yva,yte, prep = fit_transform_preprocess(X[tr],X[va],X[te],y[tr],y[va],y[te], rcfg.get("preprocess",{}), out/"preprocessing.json")
    model=build_model(X.shape[1], rcfg)
    tcfg=rcfg.get("train",{})
    model.compile(optimizer=tf.keras.optimizers.Adam(float(tcfg.get("learning_rate",1e-3))), loss=tcfg.get("loss","mse"), metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    cb=[tf.keras.callbacks.EarlyStopping(monitor=tcfg.get("monitor","val_loss"), patience=int(tcfg.get("early_stopping_patience",30)), restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(filepath=str(out/"model_best.keras"), monitor=tcfg.get("monitor","val_loss"), save_best_only=True), tf.keras.callbacks.ReduceLROnPlateau(monitor=tcfg.get("monitor","val_loss"), patience=int(tcfg.get("reduce_lr_patience",12)), factor=float(tcfg.get("reduce_lr_factor",0.5)), min_lr=float(tcfg.get("min_lr",1e-6)))]
    h=model.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=int(tcfg.get("epochs",300)),batch_size=int(tcfg.get("batch_size",32)),verbose=int(tcfg.get("verbose",1)),callbacks=cb)
    model.save(out/"model.keras")
    pd.DataFrame(h.history).to_csv(out/"history.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        if "loss" in h.history: plt.plot(h.history["loss"], label="train_loss")
        if "val_loss" in h.history: plt.plot(h.history["val_loss"], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "training_history.png", dpi=170); plt.close()
    except Exception:
        pass
    val_pred = model.predict(Xva, verbose=0).reshape(-1) * prep["y_std"] + prep["y_mean"]
    y_true = y[va]
    mae=float(np.mean(np.abs(val_pred-y_true))); rmse=float(np.sqrt(np.mean((val_pred-y_true)**2)))
    (out/"metrics_val.json").write_text(json.dumps({"mae":mae,"rmse":rmse,"n_val":int(len(va))}, indent=2), encoding="utf-8")
