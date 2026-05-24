from echopress.core.macro_windows import FirstPeakConfig, generate_first_peak_candidates, select_periodic_first_peak_sequence
import numpy as np


def test_first_peak_candidates_include_left_burst_transition():
    w = np.zeros(60)
    w[12] = 2.0
    w[18] = 5.0
    cands = generate_first_peak_candidates(w, FirstPeakConfig(k=30))
    assert 18 in cands[:3]


def test_periodicity_rejection_for_inconsistent_sequence():
    candidates = [(5, 6), (20, 21), (60, 61)]
    selected = select_periodic_first_peak_sequence(candidates, expected_k=20, tolerance=0.2)
    assert selected.indices == tuple()
    assert selected.periodicity_error > 0.2
