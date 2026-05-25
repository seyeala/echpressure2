from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class DatasetBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fft_glob: str = "**/*fft*.npz"
    feature_key: str = "features"
    target_key: str = "pressure"
    axis_key: str = "feature_axis"
    output_dir: Path = Path("ml/dataset")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_dataset(artifact_root: Path, config: DatasetBuildConfig) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    feature_axis: np.ndarray | None = None

    for artifact in sorted(Path(artifact_root).glob(config.fft_glob)):
        with np.load(artifact, allow_pickle=True) as obj:
            if config.feature_key not in obj or config.target_key not in obj:
                continue
            feats = np.asarray(obj[config.feature_key], dtype=np.float32)
            target = np.asarray(obj[config.target_key], dtype=np.float32).reshape(-1)
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
            if feats.shape[0] != target.shape[0]:
                raise ValueError(f"Feature/target length mismatch for {artifact}")
            if feature_axis is None and config.axis_key in obj:
                feature_axis = np.asarray(obj[config.axis_key])

            x_parts.append(feats)
            y_parts.append(target)
            for i in range(feats.shape[0]):
                rows.append({
                    "sample_index": len(rows),
                    "source_file": str(artifact),
                    "source_row": int(i),
                    "pressure": float(target[i]),
                })

    if not x_parts:
        raise FileNotFoundError(f"No FFT artifacts matched {config.fft_glob} under {artifact_root}")

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if feature_axis is None:
        feature_axis = np.arange(x.shape[1], dtype=np.int64)

    np.save(out_dir / "X.npy", x)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "feature_axis.npy", feature_axis)
    pd.DataFrame(rows).to_csv(out_dir / "sample_manifest.csv", index=False)

    summary = {
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "source_artifacts": int(len(x_parts)),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_yaml(out_dir / "dataset_config.resolved.yaml", config.model_dump(mode="json"))
    return summary
