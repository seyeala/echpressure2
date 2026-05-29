from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from echopress.core.config_io import load_yaml_defaults
from echopress.ml.dataset import PressureDatasetConfig, build_pressure_dataset


def test_build_pressure_dataset_with_path_config_writes_outputs(tmp_path: Path):
    pytest.importorskip("yaml", reason="PyYAML not installed")
    fft_dir = tmp_path / "fft"
    fft_dir.mkdir()
    output_dir = tmp_path / "dataset"
    config_path = tmp_path / "pressure_regression_dataset_config.yml"

    n_samples = 10
    n_features = 8
    pd.DataFrame(
        {
            "file": [f"sample_{idx}.npy" for idx in range(n_samples)],
            "path": [str(fft_dir / f"sample_{idx}.npy") for idx in range(n_samples)],
            "pressure_value": np.linspace(1.0, 10.0, n_samples),
        }
    ).to_csv(fft_dir / "fft_manifest.csv", index=False)
    np.save(fft_dir / "fft_relative_db.npy", np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features))
    np.save(fft_dir / "fft_cycles_per_window.npy", np.linspace(1.0, 8.0, n_features))

    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  feature_source: fft_relative_db",
                "  target_column: pressure_value",
                "  freq_min_cycles_per_window: 1.0",
                "  freq_max_cycles_per_window: 8.0",
                "  bin_average: 1",
                "  require_finite: true",
                "  drop_nan_targets: true",
                "split:",
                "  method: random",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  random_seed: 7",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_pressure_dataset(PressureDatasetConfig(fft_dir=fft_dir, output_dir=output_dir, config=config_path))

    expected_outputs = {
        "X.npy",
        "y.npy",
        "split.json",
        "sample_manifest.csv",
        "dataset_summary.json",
    }
    assert expected_outputs.issubset({path.name for path in output_dir.iterdir()})

    resolved = load_yaml_defaults(output_dir / "dataset_config.resolved.yml")
    assert resolved["config"] == str(config_path)
    assert isinstance(resolved["config"], str)
