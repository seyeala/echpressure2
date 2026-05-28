from pathlib import Path

import json
import numpy as np
import pandas as pd

from echopress.core.fft_export import FFTExportConfig, run_fft_postprocessed


def _write_registry(post_dir: Path, default: str = "processed_continuous_train"):
    reg = {
        "schema_version": "1.0",
        "postprocess_dir": str(post_dir),
        "default_fft_product": default,
        "products": {
            "processed_continuous_train": {
                "product_name": "processed_continuous_train", "kind": "processed",
                "path": "secondary_peak_global_periodic_continuous_train_processed_waveforms.npy",
                "manifest": "global_periodic_continuous_train_manifest.csv",
                "summary": "secondary_peak_processed_summary.json",
                "window_mode": "global-periodic-common", "window_output_layout": "continuous-train",
                "horizontal_normalized": True, "vertical_normalized": True,
                "secondary_peak_suppressed": True, "gain_normalized": True,
            },
            "raw_continuous_train": {
                "product_name": "raw_continuous_train", "kind": "raw",
                "path": "raw_global_periodic_continuous_train_waveforms.npy",
                "manifest": "global_periodic_continuous_train_manifest.csv",
                "summary": "secondary_peak_processed_summary.json",
                "window_mode": "global-periodic-common", "window_output_layout": "continuous-train",
                "horizontal_normalized": True, "vertical_normalized": False,
                "secondary_peak_suppressed": False, "gain_normalized": False,
            },
        },
    }
    (post_dir / "waveform_products.json").write_text(json.dumps(reg), encoding="utf-8")


def test_fft_source_product_processed_continuous_train(tmp_path: Path):
    post_dir = tmp_path / "post"; post_dir.mkdir()
    pd.DataFrame({"path": ["a", "b"]}).to_csv(post_dir / "global_periodic_continuous_train_manifest.csv", index=False)
    np.save(post_dir / "secondary_peak_global_periodic_continuous_train_processed_waveforms.npy", np.ones((2, 2048), dtype=np.float32))
    np.save(post_dir / "raw_global_periodic_continuous_train_waveforms.npy", np.ones((2, 2048), dtype=np.float32))
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")
    _write_registry(post_dir)

    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, source_product="processed_continuous_train", fft_mode="full", output_bins=1024))
    assert summary["source_product"] == "processed_continuous_train"
    assert summary["source_window_output_layout"] == "continuous-train"
    assert summary["n_rows"] == 2


def test_fft_source_product_raw_continuous_train(tmp_path: Path):
    post_dir = tmp_path / "post"; post_dir.mkdir()
    pd.DataFrame({"path": ["a"]}).to_csv(post_dir / "global_periodic_continuous_train_manifest.csv", index=False)
    np.save(post_dir / "secondary_peak_global_periodic_continuous_train_processed_waveforms.npy", np.ones((1, 2048), dtype=np.float32))
    np.save(post_dir / "raw_global_periodic_continuous_train_waveforms.npy", np.ones((1, 2048), dtype=np.float32))
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")
    _write_registry(post_dir)
    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, source_product="raw_continuous_train", fft_mode="full", output_bins=1024))
    assert summary["source_kind"] == "raw"


def test_fft_output_dir_source_specific(tmp_path: Path):
    post_dir = tmp_path / "post"; post_dir.mkdir()
    pd.DataFrame({"path": ["a"]}).to_csv(post_dir / "global_periodic_continuous_train_manifest.csv", index=False)
    arr=np.ones((1, 128), dtype=np.float32)
    np.save(post_dir / "secondary_peak_global_periodic_continuous_train_processed_waveforms.npy", arr)
    np.save(post_dir / "raw_global_periodic_continuous_train_waveforms.npy", arr)
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")
    _write_registry(post_dir)
    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, source_product="processed_continuous_train"))
    assert Path(summary["output_dir"]) == post_dir / "fft_outputs" / "processed_continuous_train"


def test_fft_full_does_not_crop_time_domain(tmp_path: Path):
    post_dir = tmp_path / "post"; post_dir.mkdir()
    pd.DataFrame({"path": ["a"]}).to_csv(post_dir / "global_periodic_continuous_train_manifest.csv", index=False)
    arr=np.ones((1, 4096), dtype=np.float32)
    np.save(post_dir / "secondary_peak_global_periodic_continuous_train_processed_waveforms.npy", arr)
    np.save(post_dir / "raw_global_periodic_continuous_train_waveforms.npy", arr)
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")
    _write_registry(post_dir)
    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, fft_mode="full", output_bins=1024))
    assert summary["cropped_time_domain"] is False
    assert summary["n_fft"] == summary["waveform_samples"]


def test_fft_legacy_canonical_processed(tmp_path: Path):
    post_dir = tmp_path / "post"; post_dir.mkdir()
    pd.DataFrame({"path": ["a"]}).to_csv(post_dir / "secondary_peak_processed_manifest.csv", index=False)
    np.save(post_dir / "secondary_peak_processed_waveforms.npy", np.ones((1, 128), dtype=np.float32))
    (post_dir / "secondary_peak_processed_summary.json").write_text("{}", encoding="utf-8")
    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, source_product="canonical_processed"))
    assert summary["source_product"] == "canonical_processed"
    try:
        run_fft_postprocessed(FFTExportConfig(postprocess_dir=post_dir, source_product="raw_continuous_train"))
        assert False
    except FileNotFoundError:
        assert True
