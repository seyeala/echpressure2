import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core.derivative import central_difference, local_linear, savitzky_golay


def test_derivative_estimators_accuracy():
    t = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(t)
    true = np.cos(t)
    dt = t[1] - t[0]
    W = 3

    cd = central_difference(y, dt, W)
    ll = local_linear(y, dt, W)
    sg = savitzky_golay(y, dt, W, poly_order=3)

    interior = slice(W, -W)
    assert np.allclose(cd[interior], true[interior], atol=1e-3)
    assert np.allclose(ll[interior], true[interior], atol=1e-3)
    assert np.allclose(sg[interior], true[interior], atol=1e-3)
