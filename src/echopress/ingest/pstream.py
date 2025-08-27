"""Parser for P-stream files.

P-streams provide sequences of pressure measurements ``p`` together with
associated timestamps ``T^P``.  The :func:`read_pstream` generator yields
records as :class:`PStreamRecord` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from zoneinfo import ZoneInfo
from typing import Iterator, Union, TextIO, Sequence
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
    voltages: tuple[float, float, float] | None
    pressure: float


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

    if len(parts) >= 4:
        ts_str, v1_str, v2_str, v3_str, *_ = parts
        timestamp = parse_timestamp(ts_str, settings=settings)
        voltages = (float(v1_str), float(v2_str), float(v3_str))

        alpha_arr = np.asarray(alpha, dtype=float)
        beta_arr = np.asarray(beta, dtype=float)
        pressures = alpha_arr * np.asarray(voltages) + beta_arr
        ch = settings.pressure.scalar_channel
        pressure = float(pressures[ch])
        return PStreamRecord(timestamp, voltages, pressure)

    if len(parts) >= 2:
        ts_str, p_str, *_ = parts
        timestamp = parse_timestamp(ts_str, settings=settings)
        pressure = float(p_str)
        return PStreamRecord(timestamp, None, pressure)

    raise ValueError(
        "Expected timestamp followed by three voltage columns or pressure"
    )


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
        path = pathlib.Path(path)
        if path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf8", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or not {
                    "timestamp",
                    "pressure",
                }.issubset({name.lower() for name in reader.fieldnames}):
                    raise ValueError(
                        "CSV must contain 'timestamp' and 'pressure' headers"
                    )
                tz = ZoneInfo(settings.timestamp.timezone)
                for row in reader:
                    ts_val = row.get("timestamp")
                    pr_val = row.get("pressure")
                    if ts_val is None or pr_val is None:
                        continue
                    if isinstance(ts_val, (int, float)):
                        timestamp = datetime.fromtimestamp(float(ts_val), tz=tz)
                    else:
                        token = str(ts_val).strip()
                        if token.replace(".", "", 1).isdigit():
                            timestamp = datetime.fromtimestamp(float(token), tz=tz)
                        else:
                            timestamp = parse_timestamp(token, settings=settings)
                    pressure = float(pr_val)
                    yield PStreamRecord(timestamp, None, pressure)
        else:
            with open(path, "r", encoding="utf8") as fh:
                for line in fh:
                    record = _parse_line(line, alpha, beta, settings)
                    if record is not None:
                        yield record
    else:
        for line in path:
            record = _parse_line(line, alpha, beta, settings)
            if record is not None:
                yield record
