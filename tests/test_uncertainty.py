import numpy as np

from core import pressure_uncertainty, bound_pressure


def test_pressure_uncertainty_and_bounds():
    dp_dt = np.array([1.0, -2.0, 0.5])
    e_align = np.array([0.1, 0.2, 0.3])
    kappa = 0.5

    expected = kappa * np.abs(dp_dt) * e_align
    delta = pressure_uncertainty(dp_dt, e_align, kappa)
    np.testing.assert_allclose(delta, expected)

    lower, upper = bound_pressure(dp_dt, e_align, kappa)
    np.testing.assert_allclose(lower, -expected)
    np.testing.assert_allclose(upper, expected)
    assert np.all(lower <= 0)
    assert np.all(upper >= 0)
