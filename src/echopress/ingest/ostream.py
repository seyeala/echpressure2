# src/echopress/ingest/ostream.py
"""Parser for O-stream files with optional window mode.

When ``window_mode=True`` the loader treats each file as a capture window with
start time taken from the filename stamp ``M..-D..-H..-M..-S..-U.xxx`` (year via
``base_year`` or the current UTC year).  A fixed capture duration
(``duration_s``) is used and channels are empty.  Alignment midpoint is
``start + duration_s/2``.

With ``window_mode=False`` (the default) the loader parses CSV/JSON/NPZ files
and returns their timestamps and channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import json
import re
from datetime import datetime, timezone
import numpy as np


@dataclass
class OStream:
    session_id: str
    timestamps: np.ndarray  # (N,)
    channels: np.ndarray    # (N, C)
    meta: Dict[str, Any]


_TS_ALIASES = {"timestamp", "time", "t", "ts", "Time", "Timestamp"}
_STAMP_RE = re.compile(
    r"M(?P<mon>\d{2})-D(?P<day>\d{2})-H(?P<hour>\d{2})-M(?P<minute>\d{2})-S(?P<sec>\d{2})-U\.(?P<u>\d{3})"
)


def _clean_fieldnames(fieldnames: List[str]) -> List[str]:
    return [fn.strip().lstrip("\ufeff") for fn in fieldnames]


def _parse_start_from_filename(stem: str, *, base_year: Optional[int]) -> Optional[float]:
    m = _STAMP_RE.search(stem)
    if not m:
        return None
    year = base_year if base_year is not None else datetime.now(timezone.utc).year
    dt = datetime(
        year=year,
        month=int(m.group("mon")),
        day=int(m.group("day")),
        hour=int(m.group("hour")),
        minute=int(m.group("minute")),
        second=int(m.group("sec")),
        microsecond=int(m.group("u")) * 1000,
        tzinfo=timezone.utc,
    )
    return dt.timestamp()


def load_ostream(
    path: str | Path,
    *,
    # WINDOW MODE (optional)
    duration_s: float = 0.02,
    base_year: Optional[int] = None,
    use_filename_time: bool = True,
    window_mode: bool = False,
    start_time: Optional[float] = None,
    # NON-WINDOW (fallback) options
    override_file_timestamps: bool = False,
    sampling_dt: float = 1.0,
) -> OStream:
    """Load an O-stream file (window-mode default, robust fallbacks available)."""
    path = Path(path)
    stem = path.stem

    # Resolve absolute start time
    file_start: Optional[float] = None
    if start_time is not None:
        file_start = float(start_time)
    elif use_filename_time:
        file_start = _parse_start_from_filename(stem, base_year=base_year)

    # ---- WINDOW MODE (default) ----
    if window_mode:
        if file_start is None:
            file_start = 0.0
        ts = np.array([file_start, file_start + duration_s], dtype=float)
        ch = np.zeros((2, 0), dtype=float)
        meta = {
            "mode": "window_only",
            "duration_s": duration_s,
            "start_time": file_start,
            "source": "filename_stamp" if use_filename_time else "start_time_arg",
        }
        return OStream(session_id=stem, timestamps=ts, channels=ch, meta=meta)

    # ---- NON-WINDOW FALLBACKS ----
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        session_id = data["session_id"].item() if "session_id" in data else stem
        timestamps = np.asarray(data.get("timestamps", []), dtype=float)
        channels = np.asarray(data.get("channels", []), dtype=float)
        meta = {k: data[k] for k in data.files if k not in {"session_id", "timestamps", "channels"}}
        return OStream(session_id, timestamps, channels, meta)

    if path.suffix in {".json", ".ndjson", ".txt"}:
        with open(path, "r", encoding="utf8") as fh:
            obj = json.load(fh)
        session_id = obj.get("session_id", stem)
        timestamps = np.asarray(obj.get("timestamps", []), dtype=float)
        channels = np.asarray(obj.get("channels", []), dtype=float)
        meta = {k: v for k, v in obj.items() if k not in {"session_id", "timestamps", "channels"}}
        return OStream(session_id, timestamps, channels, meta)

    if path.suffix == ".csv":
        with open(path, "r", encoding="utf8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                fns = _clean_fieldnames(reader.fieldnames)
                reader.fieldnames = fns

                if len(fns) == 1:
                    col = fns[0]
                    vals = [float(row[col]) for row in reader if row.get(col, "").strip() != ""]
                    n = len(vals)
                    if override_file_timestamps:
                        if file_start is None:
                            file_start = 0.0
                        ts = file_start + sampling_dt * np.arange(n, dtype=float)
                    else:
                        ts = np.asarray(vals, dtype=float)
                        vals = []
                    ch = (np.asarray(vals, dtype=float).reshape(n, 1) if vals else np.zeros((n, 0)))
                    return OStream(stem, ts, ch, {})

                ts_col: Optional[str] = next((c for c in fns if c in _TS_ALIASES), None)
                if ts_col is None:
                    ts_col = fns[0]
                channel_fields = [c for c in fns if c not in {ts_col, "session_id"}]
                rows = [row for row in reader if any((v or "").strip() for v in row.values())]
                session_id = rows[0].get("session_id", stem) if rows else stem

                if override_file_timestamps:
                    n = len(rows)
                    if file_start is None:
                        file_start = 0.0
                    ts = file_start + sampling_dt * np.arange(n, dtype=float)
                    ch = np.asarray([[float(r[c]) for c in channel_fields] for r in rows], dtype=float)
                    mode = "csv_headered_multi_col_override"
                else:
                    ts = np.asarray([float(r[ts_col]) for r in rows], dtype=float)
                    ch = np.asarray([[float(r[c]) for c in channel_fields] for r in rows], dtype=float)
                    mode = "csv_headered_multi_col"

                return OStream(session_id, ts, ch, {})

            # Headerless → numeric matrix
            fh.seek(0)
            raw_rows = [r for r in csv.reader(fh) if any(cell.strip() for cell in r)]
            data = np.asarray(raw_rows, dtype=float)
            if data.ndim != 2:
                raise ValueError("CSV must be 2D")

            if data.shape[1] == 1:
                n = data.shape[0]
                if override_file_timestamps:
                    if file_start is None:
                        file_start = 0.0
                    ts = file_start + sampling_dt * np.arange(n, dtype=float)
                    ch = data.astype(float)
                else:
                    ts = data[:, 0].astype(float)
                    ch = np.zeros((n, 0))
                return OStream(stem, ts, ch if ch.ndim == 2 else ch.reshape(-1, 1), {})

            ts = data[:, 0].astype(float)
            ch = data[:, 1:].astype(float)
            if override_file_timestamps:
                n = ts.shape[0]
                if file_start is None:
                    file_start = 0.0
                ts = file_start + sampling_dt * np.arange(n, dtype=float)
            return OStream(stem, ts, ch, {})

    raise ValueError(f"Unsupported O-stream file format: {path.suffix}")
