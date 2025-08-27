import numpy as np

from echopress.core import calibrate


def test_calibrate_vector_and_scalar():
    alpha = [1.0, 2.0, 3.0]
    beta = [0.0, -1.0, 1.0]
    v = np.array([1.0, 2.0, 3.0])
    pressures = calibrate(v, alpha, beta)
    expected = np.array([1.0, 3.0, 10.0])
    np.testing.assert_allclose(pressures, expected)
    scalar = calibrate(v, alpha, beta, channel=2)
    assert scalar == expected[2]
