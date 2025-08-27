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
    result = align_streams(ostream, pstream, tie_breaker="earliest", O_max=10.0, W=3, kappa=1.0)
    assert result.mapping == 0
    np.testing.assert_allclose(result.E_align, 5.0)


def test_O_max_enforcement():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([100.0, 110.0, 120.0])
    result = align_streams(
        ostream,
        pstream,
        tie_breaker="earliest",
        O_max=10.0,
        W=3,
        kappa=1.0,
    )
    assert result.mapping == -1
    assert result.diagnostics.get("rejected") is True

    result2 = align_streams(
        ostream,
        pstream,
        tie_breaker="earliest",
        O_max=10.0,
        W=3,
        kappa=1.0,
        reject_if_Ealign_gt_Omax=False,
    )
    assert result2.mapping == 0
    assert result2.diagnostics.get("E_align_violations") == [0]


def test_tie_break_behaviour():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([0.0, 10.0, 20.0])
    left = align_streams(ostream, pstream, tie_breaker="earliest", O_max=10.0, W=3, kappa=1.0)
    assert left.mapping == 0
    right = align_streams(ostream, pstream, tie_breaker="latest", O_max=10.0, W=3, kappa=1.0)
    assert right.mapping == 1


def test_alignment_with_settings():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0, 20.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([10.0, 20.0, 30.0])
    settings = Settings()
    settings.mapping.tie_breaker = "earliest"
    settings.mapping.O_max = 1.0
    settings.mapping.W = 3
    settings.mapping.kappa = 1.0
    result = align_streams(ostream, pstream, settings=settings)
    assert result.mapping == 0


def test_derivative_and_uncertainty():
    ostream = OStream(session_id="s", timestamps=np.array([0.0, 10.0, 20.0]), channels=np.zeros((0, 0)), meta={})
    pstream = make_pstream([9.0, 19.0, 29.0])
    result = align_streams(ostream, pstream, tie_breaker="earliest", O_max=10.0, W=3, kappa=0.5)
    np.testing.assert_allclose(result.diagnostics["dp_dt"], 1.0, atol=1e-6)
    np.testing.assert_allclose(result.diagnostics["delta_p"], 0.5, atol=1e-6)

