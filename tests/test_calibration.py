import numpy as np

from echopress.core.calibration import calibrate, Calibrator


def test_calibrate_vectorised():
    volts = np.array([[1.0, 2.0], [3.0, 4.0]])
    alpha = [2.0, 0.5]
    beta = [0.0, 1.0]
    pressures = calibrate(volts, alpha, beta)
    expected = np.array([[2.0, 2.0], [6.0, 3.0]])
    assert np.allclose(pressures, expected)


def test_calibrate_single_channel():
    volts = np.array([[1.0, 2.0], [3.0, 4.0]])
    alpha = [2.0, 0.5]
    beta = [0.0, 1.0]
    calib = Calibrator(alpha, beta)
    pressures = calib(volts, channel=1)
    expected = np.array([2.0, 3.0])
    assert np.allclose(pressures, expected)
