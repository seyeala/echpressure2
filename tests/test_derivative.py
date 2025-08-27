import numpy as np
import pytest

from core import central_difference, local_linear, savgol


@pytest.mark.parametrize("estimator", [central_difference, local_linear, savgol])
@pytest.mark.parametrize("W", [3, 5, 7])
def test_sine_derivative_matches_cos(estimator, W):
    t = np.linspace(0, 2 * np.pi, 1001)
    series = np.sin(t)
    dt = t[1] - t[0]
    result = estimator(series, dt, W)
    expected = np.cos(t)
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-3)
