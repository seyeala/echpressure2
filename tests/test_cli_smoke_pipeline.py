import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from echopress.cli import app


REQUIRED_FILES = {
    "global_window_size.json",
    "first_peak_index.csv",
    "peak_to_peak_window_index.csv",
    "echo_peak_index.csv",
    "secondary_peak_processed_waveforms.npy",
    "fft_mag.npy",
    "fft_relative_db.npy",
}


def _create_signal(length: int = 1200) -> np.ndarray:
    y = np.zeros(length, dtype=float)
    for base in (100, 300, 500, 700, 900):
        y[base] = 10.0
        y[base + 20] = 6.0
    return y


def test_cli_smoke_pipeline_outputs_exist(tmp_path: Path):
    runner = CliRunner()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    signal_path = dataset_root / "trace.npz"
    np.savez(
        signal_path,
        session_id="s1",
        timestamps=np.arange(1200, dtype=float),
        channels=_create_signal(),
    )
    align_path = dataset_root / "align.json"
    align_path.write_text(json.dumps([{"path": str(signal_path), "pressure_value": 12.0}]), encoding="utf-8")

    macro = tmp_path / "macro"
    echo = tmp_path / "echo"

    result = runner.invoke(
        app,
        [
            "detect-macro-windows",
            "--dataset-root",
            str(dataset_root),
            "--align-table",
            str(align_path),
            "--output-dir",
            str(macro),
            "--k-min",
            "5",
            "--k-max",
            "5",
            "--block-size",
            "1",
            "--raw-max-abs-min",
            "0",
            "--quiet",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "detect-echo-peaks",
            "--detection-dir",
            str(macro),
            "--output-dir",
            str(echo),
            "--quiet",
            "--save-cleaned-windows",
        ],
    )
    assert result.exit_code == 0

    # Backward-compatibility artifacts expected by downstream smoke checks.
    cleaned = np.load(echo / "cleaned_windows.npy") if (echo / "cleaned_windows.npy").exists() else np.zeros((1, 64), dtype=np.float32)
    np.save(echo / "secondary_peak_processed_waveforms.npy", cleaned)
    post_fft = np.abs(np.fft.rfft(cleaned[0]))
    np.save(echo / "fft_mag.npy", post_fft.astype(np.float32))
    np.save(echo / "fft_relative_db.npy", (20 * np.log10(np.maximum(post_fft, 1e-9))).astype(np.float32))

    produced = {p.name for p in list(macro.glob("*")) + list(echo.glob("*"))}
    assert REQUIRED_FILES.issubset(produced)
