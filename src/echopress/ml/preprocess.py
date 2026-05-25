from __future__ import annotations
import json
import numpy as np

def fit_transform_preprocess(X_train, X_val, X_test, y_train, y_val, y_test, cfg, out_path):
    x_mu = X_train.mean(axis=0); x_sd = X_train.std(axis=0); x_sd[x_sd==0]=1.0
    y_mu = float(np.mean(y_train)); y_sd=float(np.std(y_train) or 1.0)
    X_train=(X_train-x_mu)/x_sd; X_val=(X_val-x_mu)/x_sd; X_test=(X_test-x_mu)/x_sd
    y_train_s=(y_train-y_mu)/y_sd; y_val_s=(y_val-y_mu)/y_sd; y_test_s=(y_test-y_mu)/y_sd
    obj={"x_scaler":"standard","y_scaler":"standard","x_mean":x_mu.tolist(),"x_std":x_sd.tolist(),"y_mean":y_mu,"y_std":y_sd}
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return X_train, X_val, X_test, y_train_s, y_val_s, y_test_s, obj
