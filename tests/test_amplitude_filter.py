import json
from dataclasses import dataclass

import numpy as np

from echopress.core.amplitude_filter import (
    amplitude_metrics,
    build_low_peak_remove_list,
)


@dataclass
class DummyOStream:
    timestamps: np.ndarray
    channels: np.ndarray


def _save_alignment(tmp_path, rows):
    align = tmp_path / "align.json"
    align.write_text(json.dumps(rows))
    return align


def _touch_data(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_text("placeholder")


def test_flat_signal_gets_flagged(monkeypatch, tmp_path):
    _touch_data(tmp_path, "flat.npz")
    align = _save_alignment(
        tmp_path,
        [{"path": str(tmp_path / "flat.npz"), "sid": "s1", "file_stamp": "flat"}],
    )
    out = tmp_path / "remove.json"

    def fake_load_ostream(path, window_mode=False):
        return DummyOStream(
            timestamps=np.arange(5, dtype=float),
            channels=np.ones((5, 1), dtype=float),
        )

    monkeypatch.setattr(
        "echopress.core.amplitude_filter.load_ostream", fake_load_ostream
    )

    summary = build_low_peak_remove_list(
        align_table=align,
        dataset_root=tmp_path,
        output_list=out,
        baseline_samples=2,
        threshold_multiplier=1.0,
    )

    items = json.loads(out.read_text())
    assert summary["remove_rows"] == 1
    assert items[0]["reason"] == "max_abs_not_bigger_than_baseline_mean_abs"
    assert items[0]["max_abs"] == 1.0
    assert items[0]["baseline_mean_abs"] == 1.0


def test_signal_with_large_peak_is_kept(monkeypatch, tmp_path):
    _touch_data(tmp_path, "peak.npz")
    align = _save_alignment(
        tmp_path,
        [{"path": str(tmp_path / "peak.npz"), "sid": "s1", "file_stamp": "peak"}],
    )
    out = tmp_path / "remove.json"

    def fake_load_ostream(path, window_mode=False):
        return DummyOStream(
            timestamps=np.arange(6, dtype=float),
            channels=np.array([[1.0], [1.0], [1.0], [10.0], [1.0], [1.0]]),
        )

    monkeypatch.setattr(
        "echopress.core.amplitude_filter.load_ostream", fake_load_ostream
    )

    summary = build_low_peak_remove_list(
        align_table=align,
        dataset_root=tmp_path,
        output_list=out,
        baseline_samples=3,
        threshold_multiplier=3.0,
    )

    assert json.loads(out.read_text()) == []
    assert summary["kept_rows"] == 1
    assert summary["remove_rows"] == 0


def test_missing_file_can_be_included(tmp_path):
    align = _save_alignment(
        tmp_path,
        [{"path": str(tmp_path / "missing.npz"), "sid": "s1", "file_stamp": "missing"}],
    )
    out = tmp_path / "remove.json"

    summary = build_low_peak_remove_list(
        align_table=align,
        dataset_root=tmp_path,
        output_list=out,
        baseline_samples=2,
        include_missing=True,
    )

    items = json.loads(out.read_text())
    assert summary["missing_files"] == 1
    assert summary["remove_rows"] == 1
    assert items[0]["reason"] == "missing_file"


def test_baseline_samples_and_threshold_multiplier_behave_correctly():
    metrics = amplitude_metrics(
        np.array([2.0, -2.0, 5.0]),
        baseline_samples=2,
    )

    assert metrics["baseline_samples_used"] == 2
    assert metrics["baseline_mean_abs"] == 2.0
    assert metrics["max_abs"] == 5.0
    assert metrics["peak_to_baseline_ratio"] == 2.5
    assert metrics["max_abs"] <= 2.5 * metrics["baseline_mean_abs"]
    assert not (metrics["max_abs"] <= 2.0 * metrics["baseline_mean_abs"])
