import numpy as np
import pytest

from echopress.config import Settings
from echopress.core import central_difference, local_linear, savgol


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


def test_defaults_from_settings():
    t = np.linspace(0, 1, 11)
    series = np.sin(t)
    dt = t[1] - t[0]
    settings = Settings()
    settings.mapping.W = 5
    settings.mapping.kappa = 1.0
    expected = central_difference(series, dt, 5)
    result = central_difference(series, dt, settings=settings)
    np.testing.assert_allclose(result, expected)


def test_override_settings():
    t = np.linspace(0, 1, 11)
    series = np.sin(t)
    dt = t[1] - t[0]
    settings = Settings()
    settings.mapping.W = 7
    settings.mapping.kappa = 1.0
    result = central_difference(series, dt, W=3, settings=settings)
    expected = central_difference(series, dt, 3)
    np.testing.assert_allclose(result, expected)
