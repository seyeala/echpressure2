from pathlib import Path

import json
import numpy as np
import pandas as pd

from echopress.core.peak_window_postprocess import PeakWindowPostprocessConfig, run_peak_window_postprocess


def _make_npz(path: Path, y: np.ndarray) -> None:
    ts = np.arange(len(y), dtype=float) * 1e-6
    np.savez(path, channels=y.reshape(-1, 1), timestamps=ts)


def test_peak_to_peak_complete_windows_and_registered_not_used(tmp_path: Path):
    macro = tmp_path / "macro"
    echo = tmp_path / "echo"
    out = tmp_path / "out"
    macro.mkdir(); echo.mkdir()
    sig = np.sin(np.linspace(0, 20, 200))
    p = tmp_path / "a.npz"
    _make_npz(p, sig)

    pd.DataFrame([
        {"path": str(p), "first_peak_idx": x} for x in [10, 30, 60, 100, 150]
    ]).to_csv(macro / "first_peak_index.csv", index=False)
    pd.DataFrame([
        {"path": str(p), "first_peak_idx": x} for x in [30, 100]
    ]).to_csv(macro / "first_peak_index.registered.csv", index=False)
    (macro / "global_window_size.json").write_text(json.dumps({"T_global_samples": 40}), encoding="utf-8")

    pd.DataFrame([
        {"path": str(p), "first_peak_idx": 10, "echo_peak_offset_from_first_peak": 4},
        {"path": str(p), "first_peak_idx": 30, "echo_peak_offset_from_first_peak": 5},
    ]).to_csv(echo / "echo_peak_index.csv", index=False)

    summary = run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro, echo_dir=echo, output_dir=out))
    assert summary["n_windows"] == 4

    proc = np.load(out / "secondary_peak_processed_waveforms.npy")
    raw = np.load(out / "raw_first_peak_to_first_peak_aligned_waveforms.npy")
    manifest = pd.read_csv(out / "secondary_peak_processed_manifest.csv")
    assert proc.shape[0] == 4
    assert raw.shape[0] == 4
    assert len(manifest) == 4


def test_gain_clip_one_is_noop(tmp_path: Path):
    macro = tmp_path / "macro"
    echo = tmp_path / "echo"
    out = tmp_path / "out"
    macro.mkdir(); echo.mkdir()
    y = np.linspace(-1.0, 1.0, 80)
    p = tmp_path / "b.npz"
    _make_npz(p, y)
    pd.DataFrame([
        {"path": str(p), "first_peak_idx": 10},
        {"path": str(p), "first_peak_idx": 40},
    ]).to_csv(macro / "first_peak_index.csv", index=False)
    (macro / "global_window_size.json").write_text(json.dumps({"T_global_samples": 30}), encoding="utf-8")
    pd.DataFrame([{"path": str(p), "first_peak_idx": 10, "echo_peak_offset_from_first_peak": 2}]).to_csv(echo / "echo_peak_index.csv", index=False)

    run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro, echo_dir=echo, output_dir=out, gain_clip_min=1.0, gain_clip_max=1.0, zero_first_pulse_us=0.0, peak_neighbor_us=0.0))
    proc = np.load(out / "secondary_peak_processed_waveforms.npy")
    raw = np.load(out / "raw_first_peak_to_first_peak_aligned_waveforms.npy")
    assert np.allclose(proc, raw)


def test_fft_rows_match_processed_rows(tmp_path: Path):
    macro = tmp_path / "macro"
    echo = tmp_path / "echo"
    out = tmp_path / "out"
    fft_out = tmp_path / "fft"
    macro.mkdir(); echo.mkdir()
    p = tmp_path / "c.npz"
    _make_npz(p, np.random.default_rng(0).normal(size=120))
    pd.DataFrame([
        {"path": str(p), "first_peak_idx": 10},
        {"path": str(p), "first_peak_idx": 50},
        {"path": str(p), "first_peak_idx": 90},
    ]).to_csv(macro / "first_peak_index.csv", index=False)
    (macro / "global_window_size.json").write_text(json.dumps({"T_global_samples": 40}), encoding="utf-8")
    pd.DataFrame([{"path": str(p), "first_peak_idx": 10, "echo_peak_offset_from_first_peak": 3}]).to_csv(echo / "echo_peak_index.csv", index=False)

    run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro, echo_dir=echo, output_dir=out))

    from echopress.core.fft_export import FFTExportConfig, run_fft_postprocessed

    run_fft_postprocessed(FFTExportConfig(postprocess_dir=out, output_dir=fft_out, fft_bins=16))
    fft = np.load(fft_out / "fft_mag.npy")
    proc = np.load(out / "secondary_peak_processed_waveforms.npy")
    assert fft.shape[0] == proc.shape[0]
