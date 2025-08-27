import numpy as np
import pytest

from echopress.core.mapping import align_midpoints


def test_align_midpoints_basic():
    p = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    o = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = align_midpoints(p, o)
    # midpoints are 0.5,1.5,2.5,3.5 -> nearest indices 0..3
    assert result.indices.tolist() == [0, 1, 2, 3]
    assert pytest.approx(result.e_align) == 2.0


def test_align_midpoints_tie_break():
    p = np.array([0.0, 2.0, 4.0])
    o = np.array([0.0, 2.0, 4.0])
    # midpoints -> 1,3
    r_earlier = align_midpoints(p, o, tie_break="earlier")
    r_later = align_midpoints(p, o, tie_break="later")
    assert r_earlier.indices.tolist() == [0, 1]
    assert r_later.indices.tolist() == [1, 2]


def test_align_midpoints_o_max():
    p = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    o = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        align_midpoints(p, o, O_max=0.4)
