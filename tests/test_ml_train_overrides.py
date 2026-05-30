import sys
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from echopress.cli import app
from echopress.core.config_io import load_yaml_defaults, parse_override_value


class _FakeHistory:
    history = {"loss": [1.0], "val_loss": [1.2]}


class _FakeModel:
    def compile(self, **kwargs):
        self.compile_kwargs = kwargs

    def fit(self, *args, **kwargs):
        self.fit_kwargs = kwargs
        return _FakeHistory()

    def save(self, path):
        Path(path).write_text("fake model", encoding="utf-8")

    def predict(self, X, verbose=0):
        return np.zeros((len(X), 1), dtype=float)


def _install_fake_tensorflow(monkeypatch):
    class _Adam:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

    class _RootMeanSquaredError:
        def __init__(self, name=None):
            self.name = name

    class _Callback:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            optimizers=types.SimpleNamespace(Adam=_Adam),
            metrics=types.SimpleNamespace(RootMeanSquaredError=_RootMeanSquaredError),
            callbacks=types.SimpleNamespace(
                EarlyStopping=_Callback,
                ModelCheckpoint=_Callback,
                ReduceLROnPlateau=_Callback,
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)


def test_train_pressure_regressor_set_overrides(tmp_path, monkeypatch):
    pytest.importorskip("yaml", reason="PyYAML not installed")
    _install_fake_tensorflow(monkeypatch)

    import echopress.ml.train as train_module

    monkeypatch.setattr(train_module, "build_model", lambda input_dim, rcfg: _FakeModel())

    def fake_preprocess(Xtr, Xva, Xte, ytr, yva, yte, config, output_path):
        output_path.write_text('{"fake": true}', encoding="utf-8")
        return Xtr, Xva, Xte, ytr, yva, yte, {"y_std": 1.0, "y_mean": 0.0}

    monkeypatch.setattr(train_module, "fit_transform_preprocess", fake_preprocess)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    np.save(dataset_dir / "X.npy", np.arange(24, dtype=np.float32).reshape(6, 4))
    np.save(dataset_dir / "y.npy", np.arange(6, dtype=np.float32))
    (dataset_dir / "split.json").write_text(
        '{"train": [0, 1, 2, 3], "val": [4], "test": [5]}',
        encoding="utf-8",
    )

    base_yaml = tmp_path / "base.yml"
    base_yaml.write_text(
        "\n".join(
            [
                "model:",
                "  hidden_units: [256, 128, 64]",
                "  dropout: 0.1",
                "train:",
                "  epochs: 300",
                "  batch_size: 32",
                "  verbose: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "model"

    result = CliRunner().invoke(
        app,
        [
            "train-pressure-regressor",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--config",
            str(base_yaml),
            "--set",
            "model.hidden_units=[32,16]",
            "--set",
            "model.dropout=0.8",
            "--set",
            "train.epochs=600",
        ],
    )

    assert result.exit_code == 0, result.output
    resolved = load_yaml_defaults(output_dir / "train_config.resolved.yml")
    assert resolved["model"]["hidden_units"] == [32, 16]
    assert resolved["model"]["dropout"] == 0.8
    assert resolved["train"]["epochs"] == 600


def test_set_override_types():
    assert parse_override_value("42") == 42
    assert parse_override_value("0.125") == 0.125
    assert parse_override_value("false") is False
    assert parse_override_value("true") is True
    assert parse_override_value("[32,16]") == [32, 16]
    assert parse_override_value("null") is None
