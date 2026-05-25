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


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "pressure_stratified"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 7
    bins: int = 10
    holdout_min_pressure: float | None = None
    holdout_max_pressure: float | None = None


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _partition_indices(idxs: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(idxs)
    n_train = int(round(n * cfg.train_ratio))
    n_val = int(round(n * cfg.val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = idxs[:n_train]
    val = idxs[n_train : n_train + n_val]
    test = idxs[n_train + n_val :]
    return train, val, test


def build_split(y: np.ndarray, output_dir: Path, config: SplitConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_seed)
    indices = np.arange(len(y))

    if config.method == "random":
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        train, val, test = _partition_indices(shuffled, config)
    elif config.method == "holdout":
        if config.holdout_min_pressure is None or config.holdout_max_pressure is None:
            raise ValueError("holdout_min_pressure and holdout_max_pressure are required for holdout method")
        mask = (y >= config.holdout_min_pressure) & (y <= config.holdout_max_pressure)
        test = indices[mask]
        remainder = indices[~mask]
        rng.shuffle(remainder)
        train, val, _ = _partition_indices(remainder, config)
    else:
        quantiles = np.quantile(y, np.linspace(0.0, 1.0, config.bins + 1))
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for i in range(config.bins):
            lo, hi = quantiles[i], quantiles[i + 1]
            if i == config.bins - 1:
                bucket = indices[(y >= lo) & (y <= hi)]
            else:
                bucket = indices[(y >= lo) & (y < hi)]
            rng.shuffle(bucket)
            b_train, b_val, b_test = _partition_indices(bucket, config)
            train_parts.append(b_train)
            val_parts.append(b_val)
            test_parts.append(b_test)
        train = np.concatenate(train_parts)
        val = np.concatenate(val_parts)
        test = np.concatenate(test_parts)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    payload = {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (output_dir / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_yaml(output_dir / "split_config.resolved.yaml", config.model_dump(mode="json"))
    return payload
