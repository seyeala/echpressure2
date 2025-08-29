from __future__ import annotations

from datetime import datetime, timezone
import pathlib, re, csv
from typing import Iterator, Union, TextIO, Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------
# Timestamp grammars:
#   1) ISO-8601
#   2) HH:MM:SS[.ffffff]
#   3) float seconds since epoch
#   4) Mmm-Ddd-Hhh-Mmm-Sss-U.micro  (your custom stamp)
# ---------------------------------------------------------------------

TIMESTAMP_RE = re.compile(
    r"""^\s*(?:
            (?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)
          | (?P<hms>\d{2}:\d{2}:\d{2}(?:\.\d+)?)
          | (?P<float>\d+(?:\.\d+)?)
          | (?P<mdhmsu>M(?P<mon>\d{2})-D(?P<day>\d{2})-H(?P<hour>\d{2})-M(?P<minute>\d{2})-S(?P<sec>\d{2})-U\.(?P<u>\d{3}))
        )\s*$""",
    re.VERBOSE,
)


def _parse_values_line(line: str, *, col: int = 2) -> float:
    """
    Parse a values line like 'v1,v2,v3' or 'v1 v2 v3' and return the value at index 'col'.
    Default col=2 → 3rd value (matches your file layout).
    """
    parts = [t for t in re.split(r"[,\s]+", line.strip()) if t]
    if not parts:
        raise ValueError("Empty values line in P-stream")
    if col >= len(parts):
        raise ValueError(f"P-stream values line has {len(parts)} columns; requested col {col}")
    return float(parts[col])
    
def parse_timestamp(token: str) -> datetime:
    m = TIMESTAMP_RE.match(token)
    if not m:
        raise ValueError(f"Unrecognised timestamp: {token!r}")

    if m.group("iso"):
        iso = m.group("iso").replace("Z", "+00:00")
        return datetime.fromisoformat(iso)

    if m.group("hms"):
        fmt = "%H:%M:%S.%f" if "." in m.group("hms") else "%H:%M:%S"
        today = datetime.now(timezone.utc).date()
        return datetime.combine(today, datetime.strptime(m.group("hms"), fmt).time(), tzinfo=timezone.utc)

    if m.group("float"):
        return datetime.fromtimestamp(float(m.group("float")), tz=timezone.utc)

    if m.group("mdhmsu"):
        # Build a datetime with current year; UTC timezone
        year = datetime.now(timezone.utc).year
        mon   = int(m.group("mon"))
        day   = int(m.group("day"))
        hour  = int(m.group("hour"))
        minute= int(m.group("minute"))
        sec   = int(m.group("sec"))
        micro = int(m.group("u")) * 1000  # U.060 → 60,000 microseconds
        return datetime(year, mon, day, hour, minute, sec, micro, tzinfo=timezone.utc)

    # Should not reach here
    raise ValueError(f"Unsupported timestamp: {token!r}")

# ---------------------------------------------------------------------

@dataclass
class PStreamRecord:
    timestamp: datetime
    pressure: float

def _looks_like_timestamp_line(s: str) -> bool:
    s = s.strip()
    return bool(TIMESTAMP_RE.match(s))

def _read_pstream_text(fh: TextIO, *, value_col: int) -> Iterator[PStreamRecord]:
    pending_ts: Optional[datetime] = None
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # timestamp line?
        m = TIMESTAMP_RE.match(line)
        if m:
            pending_ts = parse_timestamp(line)
            continue
        # values line or simple "ts pressure"
        parts = [t for t in re.split(r"[,\s]+", line) if t]
        if pending_ts is not None:
            if not parts:
                raise ValueError("Empty values line after timestamp")
            if value_col >= len(parts):
                raise ValueError(f"values line has {len(parts)} columns; requested col {value_col}")
            yield PStreamRecord(pending_ts, float(parts[value_col]))
            pending_ts = None
        else:
            # simple one-line format: "<ts> <p>" or "<ts>,<p>"
            if len(parts) >= 2:
                yield PStreamRecord(parse_timestamp(parts[0]), float(parts[1]))
            else:
                raise ValueError(f"Unrecognised P-stream line: {line!r}")

def read_pstream(
    path: Union[str, pathlib.Path, TextIO],
    *,
    value_col: int = 2,   # choose 2 for your 3rd value in "v1,v2,v3"
) -> Iterator[PStreamRecord]:
    """
    Supports:
      A) Paired lines:
         M..-D..-H..-M..-S..-U.xxx
         v1,v2,v3
      B) Simple lines: "<timestamp> <pressure>" or "<timestamp>,<pressure>"
      C) True CSV with header: "timestamp,pressure"
    """
    if isinstance(path, (str, pathlib.Path)):
        p = pathlib.Path(path)
        with open(p, "r", encoding="utf8") as fh:
            # peek first non-empty, non-comment line
            first_pos = fh.tell()
            first = ""
            for line in fh:
                if line.strip() and not line.lstrip().startswith("#"):
                    first = line.rstrip("\n")
                    break
            fh.seek(first_pos)

            # Case C: true CSV header
            if "," in first and "timestamp" in first and "pressure" in first:
                reader = csv.DictReader(fh)
                for row in reader:
                    ts_raw = row["timestamp"]
                    if ts_raw.replace(".", "", 1).isdigit():
                        ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
                    else:
                        ts = parse_timestamp(ts_raw)
                    yield PStreamRecord(ts, float(row["pressure"]))
                return

            # If first line already looks like your custom timestamp → paired lines
            if _looks_like_timestamp_line(first):
                yield from _read_pstream_text(fh, value_col=value_col)
                return

            # Otherwise, try generic text parser (handles simple lines too)
            yield from _read_pstream_text(fh, value_col=value_col)
    else:
        # file-like handle
        yield from _read_pstream_text(path, value_col=value_col)
# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def read_pstream(
    path: Union[str, pathlib.Path, TextIO],
    *,
    value_col: int = 2,  # default to 3rd column in your triple
) -> Iterator[PStreamRecord]:
    """
    Yield records from a P-stream file.

    Supported layouts:
      A) Paired lines (your format):
         M08-D25-H08-M40-S42-U.060
         1.592,3.992,1.601
         (repeat...)
         - 'value_col' selects which column to output (default: 2 → third)
      B) Simple lines:
         <timestamp> <pressure>
         <timestamp>,<pressure>
      C) CSV with headers 'timestamp,pressure'  (optional)
    """
    if isinstance(path, (str, pathlib.Path)):
        p = pathlib.Path(path)
        # CSV (timestamp,pressure) path — optional support
        if p.suffix.lower() == ".csv":
            # If your CSV is actually the paired-lines format saved as .csv,
            # we still handle it via the generic text branch below.
            # Only treat as "true CSV" if it has headers timestamp,pressure.
            with open(p, "r", encoding="utf8") as fh:
                first = fh.readline()
                if "timestamp" in first and "pressure" in first:
                    fh.seek(0)
                    reader = csv.DictReader(fh)
                    for row in reader:
                        ts = parse_timestamp(row["timestamp"]) if not row["timestamp"].replace(".", "", 1).isdigit() \
                             else datetime.fromtimestamp(float(row["timestamp"]), tz=timezone.utc)
                        yield PStreamRecord(ts, float(row["pressure"]))
                    return
                # else: fall through to generic paired-line parsing using text
        # Generic text parsing (paired blocks or simple lines)
        with open(p, "r", encoding="utf8") as fh:
            yield from _read_pstream_text(fh, value_col=value_col)
    else:
        # file-like object
        yield from _read_pstream_text(path, value_col=value_col)

def _read_pstream_text(fh: TextIO, *, value_col: int) -> Iterator[PStreamRecord]:
    """Read P-stream from a text file handle supporting both paired and simple formats."""
    pending_ts: Optional[datetime] = None
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Try timestamp first
        try:
            ts = parse_timestamp(line)
            pending_ts = ts
            continue
        except Exception:
            pass

        # Not a timestamp: could be values line OR simple 'ts val'
        if pending_ts is not None:
            # Expecting a values line for the previously read timestamp
            val = _parse_values_line(line, col=value_col)
            yield PStreamRecord(pending_ts, val)
            pending_ts = None
        else:
            # Maybe it's the simple single-line format
            rec = _parse_simple_line(line)
            if rec is not None:
                yield rec
            else:
                # As last resort, try “just a number” line paired with implicit timestamp?
                raise ValueError(f"Unrecognised P-stream line (no pending timestamp): {line!r}")
