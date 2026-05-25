from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 256


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, output_dir: Path, config: EvaluateConfig) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = model.predict(X_test, batch_size=config.batch_size, verbose=0).reshape(-1)
    residuals = y_test - preds

    mse = float(np.mean((residuals) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1.0 - np.sum(residuals**2) / np.sum((y_test - y_test.mean()) ** 2))

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame({"y_true": y_test, "y_pred": preds, "residual": residuals}).to_csv(
        output_dir / "predictions.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, preds, s=10, alpha=0.6)
    lo = min(float(y_test.min()), float(preds.min()))
    hi = max(float(y_test.max()), float(preds.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True pressure")
    ax.set_ylabel("Predicted pressure")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40)
    ax.set_xlabel("Residual (true - pred)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_hist.png", dpi=150)
    plt.close(fig)

    _write_yaml(output_dir / "evaluate_config.resolved.yaml", config.model_dump(mode="json"))
    return metrics
