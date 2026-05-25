from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .models import ModelConfig, build_model

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    model: ModelConfig = ModelConfig()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    config: TrainConfig,
):
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for training") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(X_train.shape[1], config.model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=config.reduce_lr_patience),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    hist_payload = {k: [float(vv) for vv in v] for k, v in history.history.items()}
    (output_dir / "history.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")

    val_metrics = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    metrics_payload = {k: float(v) for k, v in val_metrics.items()}
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    model.save(output_dir / "model.keras")
    _write_yaml(output_dir / "train_config.resolved.yaml", config.model_dump(mode="json"))
    return model, hist_payload, metrics_payload
