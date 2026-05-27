from pathlib import Path

import numpy as np
import pandas as pd

from echopress.core.fft_export import FFTExportConfig, run_fft_postprocessed


def test_fft_export_writes_numpy_csv_and_summary(tmp_path: Path):
    post_dir = tmp_path / "post"
    out_dir = tmp_path / "fft"
    post_dir.mkdir()

    manifest = pd.DataFrame(
        {
            "path": ["a", "b", "c"],
            "first_peak_idx": [1, 2, 3],
            "first_echo_offset": [3.0, 5.0, 9.0],
        }
    )
    manifest.to_csv(post_dir / "secondary_peak_processed_manifest.csv", index=False)
    np.save(
        post_dir / "secondary_peak_processed_waveforms.npy",
        np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
    )
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")

    summary = run_fft_postprocessed(
        FFTExportConfig(postprocess_dir=post_dir, output_dir=out_dir, fft_bins=16, output_bins=16, fft_mode="full")
    )
    assert summary["output_bins"] == 16

    fft_mag = np.load(out_dir / "fft_mag.npy")
    fft_db = np.load(out_dir / "fft_db.npy")
    fft_relative_db = np.load(out_dir / "fft_relative_db.npy")
    fft_cycles = np.load(out_dir / "fft_cycles_per_window.npy")
    table = pd.read_csv(out_dir / "fft_manifest.csv")

    assert fft_mag.shape == (3, 3)
    assert fft_db.shape == fft_mag.shape
    assert fft_relative_db.shape == fft_mag.shape
    assert fft_cycles.shape == (3,)
    assert len(table) == 3


def test_fft_export_full_mode_uses_all_samples_without_time_crop(tmp_path: Path):
    post_dir = tmp_path / "post"
    out_dir = tmp_path / "fft"
    post_dir.mkdir()
    pd.DataFrame({"path": ["a"], "first_peak_idx": [1], "first_echo_offset": [3.0]}).to_csv(
        post_dir / "secondary_peak_processed_manifest.csv", index=False
    )
    waveform = np.concatenate([np.zeros(1024, dtype=np.float32), np.ones(2048, dtype=np.float32)])
    np.save(post_dir / "secondary_peak_processed_waveforms.npy", waveform.reshape(1, -1))
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")

    summary = run_fft_postprocessed(
        FFTExportConfig(postprocess_dir=post_dir, output_dir=out_dir, fft_bins=16, output_bins=16, fft_mode="full")
    )
    assert summary["waveform_samples"] == 3072
    assert summary["cropped_time_domain"] is False

    fft_mag = np.load(out_dir / "fft_mag.npy")
    assert not np.allclose(fft_mag, 0.0)


def test_fft_export_truncate_mode_keeps_legacy_behavior(tmp_path: Path):
    post_dir = tmp_path / "post"
    out_dir = tmp_path / "fft"
    post_dir.mkdir()
    pd.DataFrame({"path": ["a"], "first_peak_idx": [1], "first_echo_offset": [3.0]}).to_csv(
        post_dir / "secondary_peak_processed_manifest.csv", index=False
    )
    waveform = np.concatenate([np.zeros(16, dtype=np.float32), np.ones(32, dtype=np.float32)])
    np.save(post_dir / "secondary_peak_processed_waveforms.npy", waveform.reshape(1, -1))
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")

    summary = run_fft_postprocessed(
        FFTExportConfig(postprocess_dir=post_dir, output_dir=out_dir, fft_bins=16, fft_mode="truncate")
    )
    assert summary["cropped_time_domain"] is True
