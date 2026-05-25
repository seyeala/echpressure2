from pathlib import Path

import numpy as np
import pandas as pd

from echopress.core.fft_export import FFTExportConfig, run_fft_postprocessed


def test_fft_export_writes_numpy_csv_and_summary(tmp_path: Path):
    post_dir = tmp_path / "post"
    out_dir = tmp_path / "fft"
    post_dir.mkdir()

    pd.DataFrame(
        {
            "path": ["a", "b", "c"],
            "first_peak_idx": [1, 2, 3],
            "first_echo_offset": [3.0, 5.0, 9.0],
        }
    ).to_csv(post_dir / "postprocessed_peak_windows.csv", index=False)

    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, output_dir=out_dir, fft_bins=16))
    assert summary["fft_bins"] == 16

    arr = np.load(out_dir / "postprocessed_fft.npy")
    assert arr.ndim == 1
    assert len(arr) == 9
    table = pd.read_csv(out_dir / "postprocessed_fft.csv")
    assert len(table) == len(arr)
