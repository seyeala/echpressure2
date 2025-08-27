import numpy as np
import pytest
from datetime import datetime, timezone

from echopress.config import Settings
from echopress.ingest import OStream, PStreamRecord
from echopress.core import align_streams


def make_pstream(times):
    return [
        PStreamRecord(
            datetime.fromtimestamp(t, tz=timezone.utc),
            (0.0, 0.0, t),
            t,
        )
        for t in times
    ]


def test_basic_alignment():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0, 20.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([5.0, 15.0, 25.0])
    result = align_streams(ostream, pstream, tie_breaker="earliest", O_max=1.0, W=3, kappa=1.0)
    np.testing.assert_array_equal(result.mapping, [0, 1])
    np.testing.assert_allclose(result.E_align, [0.0, 0.0])


def test_O_max_enforcement():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([100.0, 110.0, 120.0])
    with pytest.raises(ValueError):
        align_streams(ostream, pstream, tie_breaker="earliest", O_max=10.0, W=3, kappa=1.0)


def test_tie_break_behaviour():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([0.0, 10.0, 20.0])
    left = align_streams(ostream, pstream, tie_breaker="earliest", O_max=10.0, W=3, kappa=1.0)
    assert left.mapping.tolist() == [0]
    right = align_streams(ostream, pstream, tie_breaker="latest", O_max=10.0, W=3, kappa=1.0)
    assert right.mapping.tolist() == [1]


def test_alignment_with_settings():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0, 20.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([5.0, 15.0, 25.0])
    settings = Settings(tie_breaker="earliest", O_max=1.0, W=3, kappa=1.0)
    result = align_streams(ostream, pstream, settings=settings)
    np.testing.assert_array_equal(result.mapping, [0, 1])


def test_derivative_and_uncertainty():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0, 20.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([6.0, 16.0, 26.0])
    result = align_streams(
        ostream,
        pstream,
        tie_breaker="earliest",
        O_max=10.0,
        W=3,
        method="local_linear",
        kappa=0.5,
    )
    np.testing.assert_allclose(result.diagnostics["dp_dt"], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result.diagnostics["uncertainty"], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(result.P_bounds[0], [-0.5, -0.5], atol=1e-6)
    np.testing.assert_allclose(result.P_bounds[1], [0.5, 0.5], atol=1e-6)
    assert result.diagnostics["derivative_method"] == "local_linear"
    assert result.diagnostics["window_size"] == 3

