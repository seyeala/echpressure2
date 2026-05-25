from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_x: bool = True
    scale_y: bool = True
    eps: float = 1e-8


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fit_preprocessing(X_train: np.ndarray, y_train: np.ndarray, output_dir: Path, config: PreprocessConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_mean = X_train.mean(axis=0) if config.scale_x else np.zeros(X_train.shape[1], dtype=np.float32)
    x_std = X_train.std(axis=0) if config.scale_x else np.ones(X_train.shape[1], dtype=np.float32)
    x_std = np.where(x_std < config.eps, 1.0, x_std)

    y_mean = float(y_train.mean()) if config.scale_y else 0.0
    y_std = float(y_train.std()) if config.scale_y else 1.0
    if y_std < config.eps:
        y_std = 1.0

    payload = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
        "scale_x": config.scale_x,
        "scale_y": config.scale_y,
    }
    (output_dir / "preprocessing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_yaml(output_dir / "preprocess_config.resolved.yaml", config.model_dump(mode="json"))
    return payload


def transform_with_preprocessing(X: np.ndarray, y: np.ndarray | None, preprocessing: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    X_out = X
    y_out = y
    if preprocessing.get("scale_x", True):
        x_mean = np.asarray(preprocessing["x_mean"], dtype=np.float32)
        x_std = np.asarray(preprocessing["x_std"], dtype=np.float32)
        X_out = (X_out - x_mean) / x_std
    if y_out is not None and preprocessing.get("scale_y", True):
        y_out = (y_out - float(preprocessing["y_mean"])) / float(preprocessing["y_std"])
    return X_out, y_out
