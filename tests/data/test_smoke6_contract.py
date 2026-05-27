from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIR = ROOT / "tests" / "fixtures" / "datasets" / "smoke6" / "expected"


def _load_csv(name: str) -> pd.DataFrame:
    path = EXPECTED_DIR / name
    assert path.exists(), f"Missing expected data file: {path}"
    return pd.read_csv(path)


def test_smoke6_primary_and_echo_window_counts() -> None:
    first_peak_index = _load_csv("first_peak_index.csv")

    assert "file" in first_peak_index.columns
    peaks_per_file = first_peak_index.groupby("file").size()
    assert (peaks_per_file == 5).all(), f"Expected 5 primary peaks per file, got {peaks_per_file.to_dict()}"

    peak_to_peak_windows = peaks_per_file - 1
    assert (peak_to_peak_windows == 4).all(), (
        "Expected 5 primary peaks to imply 4 complete peak-to-peak windows per file"
    )

    echo_peak_index = _load_csv("echo_peak_index.csv")
    echo_window_index = _load_csv("echo_window_index.csv")

    values_path = EXPECTED_DIR / "echo_window_values.npy"
    assert values_path.exists(), f"Missing expected data file: {values_path}"
    echo_window_values = np.load(values_path)

    assert echo_window_values.shape[0] == len(echo_window_index)
    assert len(echo_window_index) == 24
    assert len(echo_peak_index) == 72
