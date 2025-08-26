import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core.uncertainty import pressure_bound, within_bound


def test_pressure_bound_and_within_bound():
    dp_dt = np.array([1.0, -2.0, 0.5])
    e_align = 0.1
    kappa = 2.0
    expected = kappa * np.abs(dp_dt) * e_align
    bound = pressure_bound(dp_dt, e_align, kappa)
    assert np.allclose(bound, expected)

    delta_p_good = expected * 0.5
    delta_p_bad = expected * 1.5
    assert within_bound(delta_p_good, dp_dt, e_align, kappa).all()
    assert not within_bound(delta_p_bad, dp_dt, e_align, kappa).all()
