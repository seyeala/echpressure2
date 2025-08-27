"""Parser for P-stream files.

P-streams provide sequences of pressure measurements ``p`` together with
associated timestamps ``T^P``.  The :func:`read_pstream` generator yields
records as :class:`PStreamRecord` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Union, TextIO, Sequence
import pathlib
import re

import numpy as np

# Regular expression recognising the timestamp grammar.  The grammar is
# intentionally permissive in order to interoperate with a variety of
# datasets.  It accepts ISO-8601 strings, ``HH:MM:SS`` strings and plain
# floating point seconds since the Unix epoch.
TIMESTAMP_RE = re.compile(
    r"""^\s*
        (?:
            (?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)
            |
            (?P<hms>\d{2}:\d{2}:\d{2}(?:\.\d+)?)
            |
            (?P<float>\d+(?:\.\d+)?)
        )
        \s*$""",
    re.VERBOSE,
)


def parse_timestamp(token: str) -> datetime:
    """Parse a timestamp token.

    Parameters
    ----------
    token:
        String representation of a timestamp.  Supported forms include
        ISO-8601 date/time strings, ``HH:MM:SS[.ffffff]`` strings and
        numeric seconds since the Unix epoch.
    """
    m = TIMESTAMP_RE.match(token)
    if not m:
        raise ValueError(f"Unrecognised timestamp: {token!r}")

    if m.group("iso"):
        iso = m.group("iso").replace("Z", "+00:00")
        return datetime.fromisoformat(iso)

    if m.group("hms"):
        fmt = "%H:%M:%S.%f" if "." in m.group("hms") else "%H:%M:%S"
        today = datetime.now(timezone.utc).date()
        return datetime.combine(
            today, datetime.strptime(m.group("hms"), fmt).time(), tzinfo=timezone.utc
        )

    return datetime.fromtimestamp(float(m.group("float")), tz=timezone.utc)


@dataclass
class PStreamRecord:
    """Representation of a single P-stream record."""

    timestamp: datetime
    voltages: tuple[float, float, float]
    pressure: float


def _parse_line(
    line: str, alpha: Sequence[float], beta: Sequence[float]
) -> PStreamRecord | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "," in line:
        parts = [t.strip() for t in line.split(",")]
    else:
        parts = line.split()

    if len(parts) < 4:
        raise ValueError("Expected timestamp followed by three voltage columns")

    ts_str, v1_str, v2_str, v3_str, *_ = parts

    timestamp = parse_timestamp(ts_str)
    voltages = (float(v1_str), float(v2_str), float(v3_str))

    alpha_arr = np.asarray(alpha, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    pressures = alpha_arr * np.asarray(voltages) + beta_arr
    pressure = float(pressures[2])

    return PStreamRecord(timestamp, voltages, pressure)


def read_pstream(
    path: Union[str, pathlib.Path, TextIO],
    *,
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
    if alpha is None:
        alpha = (1.0, 1.0, 1.0)
    if beta is None:
        beta = (0.0, 0.0, 0.0)

    if isinstance(path, (str, pathlib.Path)):
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                record = _parse_line(line, alpha, beta)
                if record is not None:
                    yield record
    else:
        for line in path:
            record = _parse_line(line, alpha, beta)
            if record is not None:
                yield record
