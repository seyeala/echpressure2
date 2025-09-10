# src/echopress/ingest/pstream.py
"""Parser for P-stream files (paired-line friendly).

Supports:
A) Paired lines:
   M..-D..-H..-M..-S..-U.xxx
   v1,v2,v3            (or space-separated)
   -> choose 'value_col' (default 2 → third number) as pressure

B) Simple one-line:
   <timestamp> <pressure>
   <timestamp>,<pressure>

C) CSV with header:
   timestamp,pressure
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Union, TextIO, Optional, Tuple
import pathlib
import re
import csv

# Timestamp grammar (ISO / HH:MM:SS / float epoch / M..-D..-H..-M..-S..-U.xxx)
TIMESTAMP_RE = re.compile(
    r"""^\s*(?:
            (?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)
          | (?P<hms>\d{2}:\d{2}:\d{2}(?:\.\d+)?)
          | (?P<float>\d+(?:\.\d+)?)
          | (?P<mdhmsu>M(?P<mon>\d{2})-D(?P<day>\d{2})-H(?P<hour>\d{2})-M(?P<minute>\d{2})-S(?P<sec>\d{2})-U\.(?P<u>\d{3}))
        )\s*$""",
    re.VERBOSE,
)

def parse_timestamp(token: str) -> datetime:
    m = TIMESTAMP_RE.match(token)
    if not m:
        raise ValueError(f"Unrecognised timestamp: {token!r}")

    if m.group("iso"):
        return datetime.fromisoformat(m.group("iso").replace("Z", "+00:00"))
    if m.group("hms"):
        fmt = "%H:%M:%S.%f" if "." in m.group("hms") else "%H:%M:%S"
        today = datetime.now(timezone.utc).date()
        return datetime.combine(today, datetime.strptime(m.group("hms"), fmt).time(), tzinfo=timezone.utc)
    if m.group("float"):
        return datetime.fromtimestamp(float(m.group("float")), tz=timezone.utc)
    if m.group("mdhmsu"):
        year = datetime.now(timezone.utc).year
        mon, day = int(m.group("mon")), int(m.group("day"))
        hour, minute, sec = int(m.group("hour")), int(m.group("minute")), int(m.group("sec"))
        micro = int(m.group("u")) * 1000
        return datetime(year, mon, day, hour, minute, sec, micro, tzinfo=timezone.utc)

    raise ValueError(f"Unsupported timestamp: {token!r}")


@dataclass
class PStreamRecord:
    timestamp: datetime
    pressure: float
    voltages: Optional[Tuple[float, ...]] = None


class PStreamParseError(ValueError):
    """Raised when a P-stream file cannot be parsed."""

    def __init__(self, message: str, *, path: Union[str, pathlib.Path], line: int):
        self.path = str(path)
        self.line = line
        super().__init__(f"{self.path}:{self.line}: {message}")


def _parse_values_line(line: str, *, col: int = 2) -> float:
    parts = [t for t in re.split(r"[,\s]+", line.strip()) if t]
    if not parts:
        raise ValueError("Empty values line in P-stream")
    if col >= len(parts):
        raise ValueError(f"P-stream values line has {len(parts)} columns; requested col {col}")
    return float(parts[col])


def _parse_simple_line(line: str) -> Optional[PStreamRecord]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [t for t in re.split(r"[,\s]+", line) if t]
    if len(parts) >= 2:
        ts = parse_timestamp(parts[0])
        val = float(parts[1])
        return PStreamRecord(ts, val)
    return None


def _read_pstream_text(
    fh: TextIO, *, value_col: int, path: Union[str, pathlib.Path] = "<stream>"
) -> Iterator[PStreamRecord]:
    pending_ts: Optional[datetime] = None
    for lineno, raw in enumerate(fh, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            # Timestamp line?
            m = TIMESTAMP_RE.match(line)
            if m:
                pending_ts = parse_timestamp(line)
                continue

            # Values line after a timestamp
            if pending_ts is not None:
                yield PStreamRecord(pending_ts, _parse_values_line(line, col=value_col))
                pending_ts = None
            else:
                rec = _parse_simple_line(line)
                if rec is not None:
                    yield rec
                else:
                    raise ValueError(f"Unrecognised P-stream line: {line!r}")
        except ValueError as e:
            raise PStreamParseError(str(e), path=path, line=lineno) from e


def read_pstream(
    path: Union[str, pathlib.Path, TextIO],
    *,
    value_col: int = 2,
) -> Iterator[PStreamRecord]:
    """Yield PStreamRecord(timestamp, pressure) from a P-stream source."""
    if isinstance(path, (str, pathlib.Path)):
        p = pathlib.Path(path)

        # Optional CSV with header timestamp,pressure
        if p.suffix.lower() == ".csv":
            with open(p, "r", encoding="utf8") as fh:
                first_pos = fh.tell()
                first = ""
                for line in fh:
                    if line.strip() and not line.lstrip().startswith("#"):
                        first = line.rstrip("\n")
                        break
                fh.seek(first_pos)
                # Headered CSV case
                if "," in first and "timestamp" in first and "pressure" in first:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        ts_raw = row["timestamp"]
                        ts = (datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
                              if ts_raw.replace(".", "", 1).isdigit()
                              else parse_timestamp(ts_raw))
                        yield PStreamRecord(ts, float(row["pressure"]))
                    return
                # Fall back to paired/simple text parsing
                for rec in _read_pstream_text(fh, value_col=value_col, path=p):
                    yield rec
            return

        # Plain text file
        with open(p, "r", encoding="utf8") as fh:
            for rec in _read_pstream_text(fh, value_col=value_col, path=p):
                yield rec
    else:
        # File-like
        stream_name = getattr(path, "name", "<stream>")
        for rec in _read_pstream_text(path, value_col=value_col, path=stream_name):
            yield rec
