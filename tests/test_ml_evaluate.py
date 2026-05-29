import json
import sys
import types

import numpy as np
import pytest

from echopress.ml.evaluate import PressureEvalConfig, run_evaluate


pytest.importorskip("yaml")


class _FakeModel:
    def predict(self, xs, verbose=0):
        return xs[:, :1]


def test_run_evaluate_writes_split_specific_outputs(tmp_path, monkeypatch):
    def load_model(path):
        assert path.name == "model_best.keras"
        return _FakeModel()

    tf = types.ModuleType("tensorflow")
    tf.keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)

    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    dataset_dir.mkdir()
    model_dir.mkdir()

    np.save(dataset_dir / "X.npy", np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]))
    np.save(dataset_dir / "y.npy", np.array([0.0, 1.0, 1.5, 3.0, 4.5, 5.0]))
    (dataset_dir / "split.json").write_text(
        json.dumps({"train": [0, 1], "val": [2, 3], "test": [4, 5]}),
        encoding="utf-8",
    )
    (model_dir / "preprocessing.json").write_text(
        json.dumps({"x_mean": [0.0], "x_std": [1.0], "y_mean": 0.0, "y_std": 1.0}),
        encoding="utf-8",
    )

    for split in ("train", "val", "test"):
        run_evaluate(PressureEvalConfig(dataset_dir=dataset_dir, model_dir=model_dir, split=split))

        eval_dir = model_dir / f"eval_{split}"
        assert (eval_dir / f"metrics_{split}.json").exists()
        assert (eval_dir / f"predictions_{split}.csv").exists()

    assert not (model_dir / "eval_train" / "metrics_test.json").exists()
    assert not (model_dir / "eval_train" / "predictions_test.csv").exists()
    assert not (model_dir / "eval_val" / "metrics_test.json").exists()
    assert not (model_dir / "eval_val" / "predictions_test.csv").exists()
