import numpy as np

from echopress.core.window_consensus import aggregate_per_k, mad_outlier_flags, refit_with_global_k, select_global_k


def test_consensus_aggregate_and_global_k():
    agg = aggregate_per_k({20.0: [0.1, 0.2, 0.15], 40.0: [0.4, 0.5, 0.45]})
    assert select_global_k(agg) == 40.0


def test_refit_and_outlier_flags_with_weak_missing_windows():
    refit = refit_with_global_k(
        {
            "a": {40.0: 10.0},
            "b": {20.0: 7.0, 40.0: 8.0},
            "c": {20.0: 3.0},
        },
        40.0,
    )
    assert refit["a"] == 10.0
    assert refit["c"] == 3.0

    vals = np.array([1.0, 1.2, 0.8, 1.1, 6.0])
    flags = mad_outlier_flags(vals)
    assert flags[-1]
    assert not flags[0]
