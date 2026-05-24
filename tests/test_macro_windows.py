import numpy as np

from echopress.core.macro_windows import MacroConfig, build_envelope, fit_macro_k_phase


def test_macro_fit_prefers_true_period_under_micro_ambiguity():
    n = 240
    x = np.zeros(n)
    macro_peaks = np.arange(20, n, 40)
    micro_peaks = np.arange(10, n, 20)
    x[macro_peaks] = 5.0
    x[micro_peaks] += 1.0

    env = build_envelope(x, mode="max", window=5)
    fit = fit_macro_k_phase(env, MacroConfig(k_candidates=(20.0, 40.0), envelope_mode="max", envelope_window=5))
    assert fit.k == 40.0


def test_build_envelope_modes():
    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    for mode in ("rms", "max", "log_energy"):
        env = build_envelope(x, mode=mode, window=3)
        assert env.shape == x.shape
