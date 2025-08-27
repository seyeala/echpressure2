import numpy as np
import pytest

from echopress.core import align_midpoints


def test_align_midpoints_tie_breaking_and_Omax():
    p_times = np.array([0.0, 10.0])
    p_pressures = np.array([100.0, 200.0])
    o_times = np.linspace(0.0, 10.0, 11)

    result = align_midpoints(o_times, p_times, p_pressures, O_max=6, tie_breaker="earliest")
    assert result.pressure == pytest.approx(100.0)
    assert result.e_align == pytest.approx(5.0)

    result2 = align_midpoints(o_times, p_times, p_pressures, O_max=6, tie_breaker="latest")
    assert result2.pressure == pytest.approx(200.0)

    with pytest.raises(ValueError):
        align_midpoints(o_times, p_times, p_pressures, O_max=4)
