import logging
import pytest
import math

from echopress.types import Sample, TimeInterval, TimeSeries, Window
from echopress.utils.timeparse import parse_time
from echopress.utils.signals import rms, moving_average
from echopress.utils.windows import iter_windows, window_slices
from echopress.utils.logging import get_logger


def test_types():
    s: Sample = {"t": 1.0, "value": 2.0}
    assert s["t"] == 1.0
    ti = TimeInterval(0.0, 2.5)
    assert ti.duration == 2.5
    w = Window(2, 5)
    assert w.width == 3
    with pytest.raises(ValueError):
        TimeSeries([0, 1], [1])


def test_parse_time():
    assert parse_time("1:02:03.5") == pytest.approx(3723.5)
    assert parse_time("02:03") == pytest.approx(123)
    assert parse_time("45") == pytest.approx(45)
    with pytest.raises(ValueError):
        parse_time("bad")


def test_signals():
    data = [1.0, 2.0, 3.0, 4.0]
    expected = math.sqrt((1 ** 2 + 2 ** 2 + 3 ** 2 + 4 ** 2) / 4)
    assert rms(data) == pytest.approx(expected)
    assert moving_average(data, 2) == [1.5, 2.5, 3.5]
    with pytest.raises(ValueError):
        moving_average(data, 0)


def test_windows():
    data = [1, 2, 3, 4, 5]
    ws = list(iter_windows(data, 3, 2))
    assert ws == [Window(0, 3), Window(2, 5)]
    slices = window_slices(data, 3, 2)
    assert slices == [[1, 2, 3], [3, 4, 5]]


def test_logging():
    logger = get_logger("test")
    logger2 = get_logger("test")
    assert logger is logger2
    assert len(logger.handlers) == 1
    logger.debug("debug message")
