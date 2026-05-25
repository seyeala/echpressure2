from pathlib import Path

import pandas as pd

from echopress.core.peak_window_postprocess import (
    PeakWindowPostprocessConfig,
    run_peak_window_postprocess,
)


def test_postprocess_filters_peak_order_and_merges_features(tmp_path: Path):
    echo_dir = tmp_path / "echo"
    out_dir = tmp_path / "post"
    echo_dir.mkdir()

    pd.DataFrame(
        [
            {"path": "a.npz", "first_peak_idx": 10, "echo_peak_order": 1, "echo_peak_idx": 14, "echo_peak_offset_from_first_peak": 4},
            {"path": "a.npz", "first_peak_idx": 10, "echo_peak_order": 3, "echo_peak_idx": 18, "echo_peak_offset_from_first_peak": 8},
            {"path": "a.npz", "first_peak_idx": 10, "echo_peak_order": 4, "echo_peak_idx": 20, "echo_peak_offset_from_first_peak": 10},
        ]
    ).to_csv(echo_dir / "echo_peak_index.csv", index=False)

    pd.DataFrame([{"path": "a.npz", "first_peak_idx": 10, "window_start_idx": 10}]).to_csv(
        echo_dir / "echo_window_index.csv", index=False
    )

    summary = run_peak_window_postprocess(
        PeakWindowPostprocessConfig(echo_dir=echo_dir, output_dir=out_dir, max_echo_peak_order=3)
    )
    assert summary["n_windows"] == 1

    merged = pd.read_csv(out_dir / "postprocessed_peak_windows.csv")
    assert merged.loc[0, "n_echo_peaks_post"] == 2
    assert merged.loc[0, "first_echo_offset"] == 4
