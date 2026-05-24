import json

import numpy as np
import pandas as pd

from echopress.core.rmcpe import RMCPEConfig, run_rmcpe
from echopress.core.tciml import (
    TCIMLConfig,
    _expected_centers,
    _ncc,
    run_tciml,
)


def _pulse_signal(length, centers, width=3, amp=1.0, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    sig = rng.normal(0.0, noise, size=length)
    x = np.arange(length)
    for c in centers:
        sig += amp * np.exp(-0.5 * ((x - c) / width) ** 2)
    return sig


def test_expected_centers_and_complete_window_bounds_filtering():
    cfg = TCIMLConfig(T_hat=20.0, T_error_samples=1.0, peak_width_samples=5)
    centers = _expected_centers(N=51, phase=3.0, cfg=cfg)

    # First/last phase-aligned centers are dropped when full [c-w, c+w] window spills out of bounds.
    np.testing.assert_array_equal(centers, np.array([23, 43]))


def test_normalized_ncc_peak_localization_near_expected_phase():
    template = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    signal = np.zeros(60)
    signal[30:35] = template

    ncc = _ncc(signal, template)
    peak = int(np.argmax(ncc))

    assert abs((peak + template.size // 2) - 32) <= 1


def test_ncc_regression_matches_prior_loop_behavior():
    rng = np.random.default_rng(123)
    signal = rng.normal(size=48)
    template = rng.normal(size=7)

    def old_ncc(sig, tpl):
        t = (tpl - np.mean(tpl)) / np.linalg.norm(tpl - np.mean(tpl))
        out = np.zeros(sig.size - tpl.size + 1, dtype=float)
        for i in range(out.size):
            seg = sig[i : i + tpl.size]
            seg = seg - np.mean(seg)
            norm = np.linalg.norm(seg)
            seg = np.zeros_like(seg) if norm <= 1e-12 else seg / norm
            out[i] = float(np.dot(seg, t))
        return out

    np.testing.assert_allclose(_ncc(signal, template), old_ncc(signal, template), rtol=1e-12, atol=1e-12)


def test_onset_end_extraction_threshold_fallback_behavior(tmp_path):
    sig = np.zeros(160)
    sig[40] = 4.0
    sig[80] = 4.0
    files = [sig]

    cfg = TCIMLConfig(
        T_hat=40.0,
        T_error_samples=1.0,
        peak_width_samples=2,
        W_minus=6,
        W_plus=7,
        envelope_rel_threshold=0.0,
        C_min=-1.0,
    )
    period_df = pd.DataFrame([{"file_id": "file_0", "accepted": True, "T_i": 40.0, "score": 1.0}])

    old = tmp_path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        out = run_tciml(files, {}, period_df, cfg)
    finally:
        os.chdir(old)

    assert not out.empty
    row = out.iloc[0]
    assert row["onset_idx"] == row["matched_center_idx"] - cfg.W_minus
    assert row["end_idx"] == row["matched_center_idx"] + cfg.W_plus


def test_supplied_phase_used_when_enabled():
    files = [_pulse_signal(120, centers=[35, 65, 95], width=1, amp=5.0, noise=0.0)]
    cfg = TCIMLConfig(T_hat=30.0, T_error_samples=1.0, peak_width_samples=2, C_min=-1.0, use_supplied_phase=True)
    period_df = pd.DataFrame([{"file_id": "file_0", "T_i": 30.0, "phase_i": 5.0, "accepted": True, "score": 1.0}])
    out = run_tciml(files, {"T_hat": 30.0}, period_df, cfg, write_artifacts=False)
    assert not out.empty
    assert (out["expected_center_idx"] == 35).any()


def test_fallback_phase_used_when_disabled_or_missing():
    files = [_pulse_signal(120, centers=[35, 65, 95], width=1, amp=5.0, noise=0.0)]
    supplied = pd.DataFrame([{"file_id": "file_0", "T_i": 30.0, "phase_i": 5.0, "accepted": True, "score": 1.0}])
    missing = pd.DataFrame([{"file_id": "file_0", "T_i": 30.0, "phase_i": np.nan, "accepted": True, "score": 1.0}])

    cfg_disabled = TCIMLConfig(T_hat=30.0, T_error_samples=1.0, peak_width_samples=2, C_min=-1.0, use_supplied_phase=False)
    out_disabled = run_tciml(files, {"T_hat": 30.0}, supplied, cfg_disabled, write_artifacts=False)
    assert (out_disabled["expected_center_idx"] == 35).any()

    cfg_enabled = TCIMLConfig(T_hat=30.0, T_error_samples=1.0, peak_width_samples=2, C_min=-1.0, use_supplied_phase=True)
    out_missing = run_tciml(files, {"T_hat": 30.0}, missing, cfg_enabled, write_artifacts=False)
    assert (out_missing["expected_center_idx"] == 35).any()


def test_marker_rejection_reasons_low_score_and_high_residual(tmp_path):
    sig = np.zeros(220)
    sig[120] = 5.0
    files = [sig]
    cfg = TCIMLConfig(
        T_hat=40.0,
        T_error_samples=1.0,
        peak_width_samples=3,
        C_min=1.1,
        P_min=0.0,
        search_radius_min=8,
    )
    period_df = pd.DataFrame([{"file_id": "file_0", "accepted": True, "T_i": 40.0, "score": 1.0}])

    old = tmp_path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        out = run_tciml(files, {}, period_df, cfg)
    finally:
        os.chdir(old)

    assert not out.empty
    assert (~out["accepted"]).any()
    assert out["reject_reason"].str.contains("score_low").any()
    assert out["reject_reason"].str.contains("residual_large").any()


def test_output_dir_and_write_artifacts_controls(tmp_path):
    files = [_pulse_signal(120, centers=[20, 50, 80], width=1, amp=4.0, noise=0.0)]
    cfg = TCIMLConfig(T_hat=30.0, T_error_samples=1.0, peak_width_samples=2, C_min=-1.0)
    period_df = pd.DataFrame([{"file_id": "file_0", "T_i": 30.0, "phase_i": 20.0, "accepted": True, "score": 1.0}])
    out_dir = tmp_path / "artifacts"

    run_tciml(files, {"T_hat": 30.0}, period_df, cfg, output_dir=out_dir, write_artifacts=True)
    assert (out_dir / "incident_template.npy").exists()
    assert (out_dir / "incident_marker_table.csv").exists()
    assert (out_dir / "incident_marker_summary.json").exists()

    suppressed_dir = tmp_path / "suppressed"
    run_tciml(files, {"T_hat": 30.0}, period_df, cfg, output_dir=suppressed_dir, write_artifacts=False)
    assert not (suppressed_dir / "incident_template.npy").exists()
    assert not (suppressed_dir / "incident_marker_table.csv").exists()
    assert not (suppressed_dir / "incident_marker_summary.json").exists()


def test_serialization_schema_columns_match_documented_names(tmp_path):
    period = 30
    files = [
        _pulse_signal(300, centers=range(20, 280, period), amp=3.0, seed=101, noise=0.01),
        _pulse_signal(300, centers=range(25, 285, period), amp=3.0, seed=202, noise=0.01),
    ]

    old = tmp_path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        rmcpe_cfg = RMCPEConfig(T_min=20.0, T_max=40.0, prominence=0.1, random_seed=13, bootstrap_count=50)
        summary, per_file = run_rmcpe(files, rmcpe_cfg)
        tciml_cfg = TCIMLConfig(T_hat=summary["T_hat"], T_error_samples=summary["T_error_samples"], peak_width_samples=3)
        run_tciml(files, summary, per_file, tciml_cfg)

        summary_json = json.loads((tmp_path / "window_period_summary.json").read_text())
        marker_meta = json.loads((tmp_path / "incident_marker_summary.json").read_text())
        period_csv = pd.read_csv(tmp_path / "window_period_per_file.csv")
        marker_csv = pd.read_csv(tmp_path / "incident_marker_table.csv")
    finally:
        os.chdir(old)

    assert set(summary_json) >= {"algorithm", "config", "n_files", "n_accepted", "T_hat", "T_error_samples", "T_bootstrap_ci"}
    assert set(marker_meta) >= {"algorithm", "config", "n_files", "n_markers", "n_accepted"}
    assert set(period_csv.columns) == {
        "file_id",
        "accepted",
        "reject_reason",
        "T_i",
        "phase_i",
        "score",
        "residual_median",
        "residual_mad",
        "n_peaks",
    }
    assert set(marker_csv.columns) == {
        "file_id",
        "expected_center_idx",
        "matched_center_idx",
        "window_start_idx",
        "window_end_idx",
        "onset_idx",
        "end_idx",
        "raw_ncc",
        "env_ncc",
        "blended_score",
        "residual_samples",
        "local_prominence",
        "local_amplitude",
        "accepted",
        "reject_reason",
    }
