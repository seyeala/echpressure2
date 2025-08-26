"""Parser for P-stream files.

P-streams provide sequences of pressure measurements ``p`` together with
associated timestamps ``T^P``.  The :func:`read_pstream` generator yields
records as :class:`PStreamRecord` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Union, TextIO
import pathlib
import re

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
    pressure: float


def _parse_line(line: str) -> PStreamRecord | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "," in line:
        ts_str, p_str, *_ = line.split(",")
    else:
        ts_str, p_str, *_ = line.split()

    timestamp = parse_timestamp(ts_str)
    pressure = float(p_str)
    return PStreamRecord(timestamp, pressure)


def read_pstream(path: Union[str, pathlib.Path, TextIO]) -> Iterator[PStreamRecord]:
    """Yield records from a P-stream file.

    Parameters
    ----------
    path:
        Either a path-like object or a text file object providing lines.
    """
    if isinstance(path, (str, pathlib.Path)):
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                record = _parse_line(line)
                if record is not None:
                    yield record
    else:
        for line in path:
            record = _parse_line(line)
            if record is not None:
                yield record
