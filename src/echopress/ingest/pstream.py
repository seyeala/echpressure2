"""Parser for P-stream files.

P-streams provide sequences of pressure measurements ``p`` together with
associated timestamps ``T^P``.  The :func:`read_pstream` generator yields
records as :class:`PStreamRecord` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Iterator, Union, TextIO, Sequence
import csv
import pathlib
import re

import numpy as np

from ..config import Settings

# Regular expression recognising the timestamp grammar.  The grammar is
# intentionally permissive in order to interoperate with a variety of
# datasets.  It accepts ISO-8601 strings, ``HH:MM:SS`` strings,
# ``Mxx-Dxx-Hxx-Mxx-Sxx-U.xxx`` strings and plain floating point seconds
# since the Unix epoch.
TIMESTAMP_RE = re.compile(
    r"""^\s*
        (?:
            (?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)
            |
            (?P<hms>\d{2}:\d{2}:\d{2}(?:\.\d+)?)
            |
            (?P<float>\d+(?:\.\d+)?)
            |
            M(?P<month>\d{2})-D(?P<day>\d{2})-H(?P<hour>\d{2})-M(?P<minute>\d{2})-S(?P<second>\d{2})-U(?:\.(?P<subsecond>\d+))?
        )
        \s*$""",
    re.VERBOSE,
)


def parse_timestamp(token: str, *, settings: Settings | None = None) -> datetime:
    """Parse a timestamp token using ``settings.timestamp`` controls."""

    if settings is None:
        settings = Settings()

    ts_cfg = settings.timestamp
    tz = ZoneInfo(ts_cfg.timezone)

    if ts_cfg.format:
        try:
            dt = datetime.strptime(token, ts_cfg.format)
            return dt.replace(tzinfo=tz)
        except ValueError:
            pass

    m = TIMESTAMP_RE.match(token)
    if not m:
        raise ValueError(f"Unrecognised timestamp: {token!r}")

    if m.group("iso"):
        iso = m.group("iso").replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt

    if m.group("hms"):
        fmt = "%H:%M:%S.%f" if "." in m.group("hms") else "%H:%M:%S"
        today = datetime.now(tz).date()
        date = today.replace(year=ts_cfg.year_fallback)
        return datetime.combine(
            date,
            datetime.strptime(m.group("hms"), fmt).time(),
            tzinfo=tz,
        )

    if m.group("float"):
        return datetime.fromtimestamp(float(m.group("float")), tz=tz)

    sub = m.group("subsecond") or ""
    microsecond = int((sub + "000000")[:6])
    return datetime(
        ts_cfg.year_fallback,
        int(m.group("month")),
        int(m.group("day")),
        int(m.group("hour")),
        int(m.group("minute")),
        int(m.group("second")),
        microsecond,
        tzinfo=tz,
    )


@dataclass
class PStreamRecord:
    """Representation of a single P-stream record."""

    timestamp: datetime
    pressure: float
    voltages: tuple[float, float, float] | None = None


def _parse_line(
    line: str,
    alpha: Sequence[float],
    beta: Sequence[float],
    settings: Settings,
) -> PStreamRecord | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "," in line:
        parts = [t.strip() for t in line.split(",")]
    else:
        parts = line.split()

    if len(parts) < 2:
        raise ValueError(
            "Expected timestamp and pressure or timestamp followed by three voltage columns"
        )

    ts_str = parts[0]
    timestamp = parse_timestamp(ts_str, settings=settings)

    if len(parts) >= 4:
        v1_str, v2_str, v3_str = parts[1:4]
        voltages = (float(v1_str), float(v2_str), float(v3_str))

        alpha_arr = np.asarray(alpha, dtype=float)
        beta_arr = np.asarray(beta, dtype=float)
        pressures = alpha_arr * np.asarray(voltages) + beta_arr
        ch = settings.pressure.scalar_channel
        pressure = float(pressures[ch])
    else:
        p_str = parts[1]
        voltages = None
        pressure = float(p_str)

    return PStreamRecord(timestamp, pressure, voltages)


def read_pstream(
    path: Union[str, pathlib.Path, TextIO],
    *,
    settings: Settings | None = None,
    alpha: Sequence[float] | None = None,
    beta: Sequence[float] | None = None,
) -> Iterator[PStreamRecord]:
    """Yield records from a P-stream file.

    Parameters
    ----------
    path:
        Either a path-like object or a text file object providing lines.
    alpha, beta:
        Per-channel calibration coefficients. If omitted an identity
        calibration is used.
    """
    if settings is None:
        settings = Settings()

    if alpha is None:
        alpha = settings.calibration.alpha
    if beta is None:
        beta = settings.calibration.beta

    if isinstance(path, (str, pathlib.Path)):
        p = pathlib.Path(path)
        if p.suffix.lower() == ".csv":
            tz = ZoneInfo(settings.timestamp.timezone)
            with open(p, newline="", encoding="utf8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    ts_val = row.get("timestamp")
                    p_val = row.get("pressure")
                    if ts_val is None or p_val is None:
                        continue
                    try:
                        timestamp = parse_timestamp(str(ts_val), settings=settings)
                    except ValueError:
                        timestamp = datetime.fromtimestamp(float(ts_val), tz=tz)
                    pressure = float(p_val)
                    yield PStreamRecord(timestamp, pressure, None)
        else:
            with open(p, "r", encoding="utf8") as fh:
                for line in fh:
                    record = _parse_line(line, alpha, beta, settings)
                    if record is not None:
                        yield record
    else:
        for line in path:
            record = _parse_line(line, alpha, beta, settings)
            if record is not None:
                yield record
