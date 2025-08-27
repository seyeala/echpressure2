import numpy as np
import pytest

from echopress.core import CalibrationCoefficients, apply_calibration


def test_apply_calibration_multiple_channels():
    voltage = np.array([0.0, 1.0, 2.0])
    coeffs = CalibrationCoefficients(alpha=np.array([1.0, 2.0, -1.0]),
                                     beta=np.array([0.0, -1.0, 0.5]))

    expected = [1.0 * voltage + 0.0,
                2.0 * voltage - 1.0,
                -1.0 * voltage + 0.5]

    for ch, exp in enumerate(expected):
        np.testing.assert_allclose(apply_calibration(voltage, coeffs, ch), exp)


def test_apply_calibration_broadcasting():
    voltage = np.ones((2, 3))
    coeffs = CalibrationCoefficients(alpha=np.array([2.0]),
                                     beta=np.array([1.0]))
    expected = 2.0 * voltage + 1.0
    result = apply_calibration(voltage, coeffs, 0)
    np.testing.assert_allclose(result, expected)


def test_mismatched_coefficient_lengths():
    with pytest.raises(ValueError):
        CalibrationCoefficients(alpha=np.array([1.0, 2.0]),
                                beta=np.array([0.0]))
