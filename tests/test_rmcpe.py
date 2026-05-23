import json

import numpy as np

from echopress.core.rmcpe import (
    RMCPEConfig,
    _block_max_envelope,
    _fit_file,
)


def _impulse_train(length, period, phase=0, amplitude=1.0, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    sig = rng.normal(0.0, noise, size=length)
    for idx in range(phase, length, period):
        sig[idx] += amplitude
    return sig


def test_block_max_envelope_index_mapping_correctness():
    signal = np.array([0.0, -2.0, 1.0, 5.0, -4.0, 0.5, 0.0, 3.0])
    env, idx_map, block_size = _block_max_envelope(signal, max_points=4)

    assert block_size == 2
    np.testing.assert_allclose(env, [2.0, 5.0, 4.0, 3.0])
    np.testing.assert_array_equal(idx_map, [1, 3, 4, 7])


def test_harmonic_spacing_recovery_under_missed_peaks():
    period = 40
    length = 900
    sig = _impulse_train(length, period, phase=10, amplitude=5.0, seed=7, noise=0.05)
    # Miss every 4th true peak to force l-harmonic spacing recovery.
    for idx in range(10, length, period * 4):
        sig[idx] = 0.0

    cfg = RMCPEConfig(T_min=30.0, T_max=55.0, prominence=0.2, random_seed=123)
    res = _fit_file("fileA", sig, cfg)

    assert res.accepted
    assert res.T_i is not None
    assert abs(res.T_i - period) <= 1.0


def test_median_mad_robustness_with_outlier_spacings():
    period = 50
    peaks = np.array([5, 55, 105, 155, 205, 255, 505, 555], dtype=int)
    sig = np.zeros(700, dtype=float)
    sig[peaks] = 10.0

    cfg = RMCPEConfig(T_min=40.0, T_max=70.0, prominence=0.1)
    res = _fit_file("fileB", sig, cfg)

    assert res.accepted
    assert res.T_i is not None
    assert abs(res.T_i - period) <= 1.0
    assert res.residual_mad is not None
    assert res.residual_mad <= 1.0


def test_comb_score_prefers_true_period_over_2x_alias():
    period = 24
    sig = _impulse_train(1200, period, phase=5, amplitude=8.0, seed=11, noise=0.02)

    cfg_true = RMCPEConfig(T_min=20.0, T_max=30.0, prominence=0.2)
    cfg_alias = RMCPEConfig(T_min=42.0, T_max=54.0, prominence=0.2)
    true_res = _fit_file("true", sig, cfg_true)
    alias_res = _fit_file("alias", sig, cfg_alias)

    assert true_res.accepted
    assert alias_res.accepted
    assert true_res.score > alias_res.score
