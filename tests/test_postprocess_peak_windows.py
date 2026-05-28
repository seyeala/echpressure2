from pathlib import Path

import json
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from echopress.cli import app
from echopress.core.peak_window_postprocess import (
    PeakWindowPostprocessConfig,
    build_global_periodic_window_plan,
    run_peak_window_postprocess,
)


def _make_npz(path: Path, y: np.ndarray) -> None:
    ts = np.arange(len(y), dtype=float) * 1e-6
    np.savez(path, channels=y.reshape(-1, 1), timestamps=ts)


def test_build_global_periodic_window_plan_forward_common_count():
    first_df = pd.DataFrame([
        {"path": "A", "first_peak_idx": 10, "file": "A", "file_index": 0, "pressure_value": 1.0},
        {"path": "A", "first_peak_idx": 110, "file": "A", "file_index": 0, "pressure_value": 1.0},
        {"path": "A", "first_peak_idx": 210, "file": "A", "file_index": 0, "pressure_value": 1.0},
        {"path": "A", "first_peak_idx": 310, "file": "A", "file_index": 0, "pressure_value": 1.0},
        {"path": "B", "first_peak_idx": 12, "file": "B", "file_index": 1, "pressure_value": 2.0},
        {"path": "B", "first_peak_idx": 112, "file": "B", "file_index": 1, "pressure_value": 2.0},
        {"path": "B", "first_peak_idx": 212, "file": "B", "file_index": 1, "pressure_value": 2.0},
    ])
    plan, summary = build_global_periodic_window_plan(first_df, {"A": 420, "B": 330}, 100, 0.12, anchor="first")
    assert summary["common_window_count"] == 3
    assert len(plan) == 6
    assert (plan["window_len_samples"] == 100).all()
    assert (plan["end_idx_exclusive"] == plan["start_first_peak_idx"] + 100).all()


def test_build_global_periodic_window_plan_rejects_bad_periodicity():
    first_df = pd.DataFrame([
        {"path": "A", "first_peak_idx": 10}, {"path": "A", "first_peak_idx": 110}, {"path": "A", "first_peak_idx": 250},
        {"path": "B", "first_peak_idx": 12}, {"path": "B", "first_peak_idx": 112}, {"path": "B", "first_peak_idx": 212},
    ])
    plan, summary = build_global_periodic_window_plan(first_df, {"A": 420, "B": 330}, 100, 0.12, anchor="first")
    assert summary["common_window_count"] == 2
    assert len(plan[plan["path"] == "A"]) == 2


def _prep_fixture(tmp_path: Path):
    macro = tmp_path / "macro"; echo = tmp_path / "echo"; out = tmp_path / "out"
    macro.mkdir(); echo.mkdir()
    p1 = tmp_path / "a.npz"; p2 = tmp_path / "b.npz"
    _make_npz(p1, np.sin(np.linspace(0, 20, 420)))
    _make_npz(p2, np.cos(np.linspace(0, 20, 330)))
    pd.DataFrame([
        {"path": str(p1), "first_peak_idx": 10, "file": "a", "file_index": 0, "pressure_value": 1.0},
        {"path": str(p1), "first_peak_idx": 110, "file": "a", "file_index": 0, "pressure_value": 1.0},
        {"path": str(p1), "first_peak_idx": 210, "file": "a", "file_index": 0, "pressure_value": 1.0},
        {"path": str(p2), "first_peak_idx": 12, "file": "b", "file_index": 1, "pressure_value": 2.0},
        {"path": str(p2), "first_peak_idx": 112, "file": "b", "file_index": 1, "pressure_value": 2.0},
        {"path": str(p2), "first_peak_idx": 212, "file": "b", "file_index": 1, "pressure_value": 2.0},
    ]).to_csv(macro / "first_peak_index.csv", index=False)
    (macro / "global_window_size.json").write_text(json.dumps({"T_global_samples": 100}), encoding="utf-8")
    pd.DataFrame([
        {"path": str(p1), "first_peak_idx": 10, "echo_peak_offset_from_first_peak": 3},
        {"path": str(p2), "first_peak_idx": 12, "echo_peak_offset_from_first_peak": 4},
    ]).to_csv(echo / "echo_peak_index.csv", index=False)
    return macro, echo, out


def test_postprocess_global_periodic_common_shape(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    summary = run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro, echo_dir=echo, output_dir=out, window_mode="global-periodic-common"))
    assert (out / "secondary_peak_global_periodic_processed_waveforms.npy").exists()
    assert (out / "raw_global_periodic_aligned_waveforms.npy").exists()
    arr = np.load(out / "secondary_peak_global_periodic_processed_waveforms.npy")
    manifest = pd.read_csv(out / "global_periodic_window_manifest.csv")
    assert arr.shape[1] == 100
    assert len(manifest) == arr.shape[0]
    assert summary["window_mode"] == "global-periodic-common"


def test_postprocess_peak_to_peak_backward_compatible(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro, echo_dir=echo, output_dir=out))
    assert (out / "raw_first_peak_to_first_peak_aligned_waveforms.npy").exists()
    assert (out / "secondary_peak_processed_waveforms.npy").exists()


def test_cli_postprocess_global_periodic_common(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["postprocess-peak-windows", "--macro-dir", str(macro), "--echo-dir", str(echo), "--output-dir", str(out), "--window-mode", "global-periodic-common", "--window-anchor", "first", "--periodicity-tolerance-frac", "0.12"])
    assert result.exit_code == 0
    summary = json.loads((out / "secondary_peak_processed_summary.json").read_text(encoding="utf-8"))
    assert summary["window_mode"] == "global-periodic-common"
    assert summary["waveform_shape"][1] == summary["T_global_samples"]


def test_postprocess_global_periodic_continuous_train_shape(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    summary = run_peak_window_postprocess(PeakWindowPostprocessConfig(
        macro_dir=macro, echo_dir=echo, output_dir=out, window_mode="global-periodic-common", window_output_layout="continuous-train"
    ))
    arr = np.load(out / "secondary_peak_global_periodic_continuous_train_processed_waveforms.npy")
    assert arr.shape == (2, 300)
    assert summary["window_output_layout"] == "continuous-train"
    assert summary["waveform_shape"] == [2, 300]
    assert summary["train_samples"] == 300
    assert (out / "global_periodic_continuous_train_manifest.csv").exists()
    assert (out / "raw_global_periodic_continuous_train_waveforms.npy").exists()


def test_cli_postprocess_global_periodic_continuous_train(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, [
        "postprocess-peak-windows", "--macro-dir", str(macro), "--echo-dir", str(echo), "--output-dir", str(out),
        "--window-mode", "global-periodic-common", "--window-anchor", "first", "--window-output-layout", "continuous-train"
    ])
    assert result.exit_code == 0
    summary = json.loads((out / "secondary_peak_processed_summary.json").read_text(encoding="utf-8"))
    assert summary["window_output_layout"] == "continuous-train"
    assert summary["waveform_shape"] == [2, 300]


def test_plan_only(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    summary = run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro, echo_dir=echo, output_dir=out, window_mode="global-periodic-common", plan_only=True))
    assert (out / "global_periodic_window_plan.csv").exists()
    assert (out / "secondary_peak_processed_summary.json").exists()
    assert not (out / "secondary_peak_global_periodic_processed_waveforms.npy").exists()
    assert summary["plan_only"] is True


def test_postprocess_writes_waveform_products_continuous_train(tmp_path: Path):
    macro, echo, out = _prep_fixture(tmp_path)
    run_peak_window_postprocess(PeakWindowPostprocessConfig(
        macro_dir=macro, echo_dir=echo, output_dir=out, window_mode="global-periodic-common", window_output_layout="continuous-train"
    ))
    registry = json.loads((out / "waveform_products.json").read_text(encoding="utf-8"))
    products = registry["products"]
    assert "processed_continuous_train" in products
    assert "raw_continuous_train" in products
    summary = json.loads((out / "secondary_peak_processed_summary.json").read_text(encoding="utf-8"))
    assert products["processed_continuous_train"]["shape"] == summary["waveform_shape"]
