"""Voltage to pressure calibration routines."""

from __future__ import annotations

from typing import Iterable, Sequence, Union

from ..config import get_config

Number = Union[int, float]


def _apply_linear(v: Number, a: Number, b: Number) -> Number:
    """Apply a simple linear transform."""
    return a * v + b


def calibrate(
    voltage: Union[Number, Sequence[Number]],
    channel: int | None = None,
    *,
    alpha: Sequence[Number] | None = None,
    beta: Sequence[Number] | None = None,
) -> Union[Number, list[Number]]:
    """Map voltages to pressure values using calibration coefficients.

    Parameters
    ----------
    voltage:
        Voltage reading or sequence of readings.
    channel:
        Channel index.  If omitted, :mod:`echopress`' configuration
        ``scalar_channel`` setting is used.
    alpha, beta:
        Optional explicit coefficient sequences overriding the configured
        values.

    Returns
    -------
    pressure:
        A single calibrated pressure value or a list of values matching
        the input sequence.
    """

    cfg = get_config()
    alpha = list(alpha) if alpha is not None else cfg.alpha
    beta = list(beta) if beta is not None else cfg.beta
    if channel is None:
        channel = cfg.scalar_channel

    try:
        a = alpha[channel]
        b = beta[channel]
    except IndexError as exc:  # pragma: no cover - defensive branch
        raise ValueError("Channel index out of range") from exc

    if isinstance(voltage, Iterable) and not isinstance(voltage, (str, bytes)):
        return [_apply_linear(v, a, b) for v in voltage]
    return _apply_linear(voltage, a, b)
