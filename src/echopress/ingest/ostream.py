"""Parser for O-stream files.

O-streams typically contain multi-channel observation arrays, per-sample
timestamps and metadata such as session identifiers. This module provides
:func:`load_ostream` to load these files into the :class:`OStream` data class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv
import json
import re
from datetime import datetime, timezone

import numpy as np


@dataclass
class OStream:
    """In-memory representation of an O-stream."""
    session_id: str
    timestamps: np.ndarray  # shape (N,)
    channels: np.ndarray    # shape (N, C)
    meta: Dict[str, Any]


# Accept common aliases for timestamp column names in headered CSVs
_TS_ALIASES = {"timestamp", "time", "t", "ts", "Time", "Timestamp"}

# Filename pattern: ...M08-D25-H08-M40-S45-U.334...
_STAMP_RE = re.compile(
    r"M(?P<mon>\d{2})-D(?P<day>\d{2})-H(?P<hour>\d{2})-M(?P<minute>\d{2})-S(?P<sec>\d{2})-U\.(?P<u>\d{3})"
)


def _clean_fieldnames(fieldnames: List[str]) -> List[str]:
    return [fn.strip().lstrip("\ufeff") for fn in fieldnames]


def _parse_start_from_filename(stem: str, *, base_year: Optional[int]) -> Optional[float]:
    """
    Parse M..-D..-H..-M..-S..-U.xxx from filename stem and return epoch seconds.
    If base_year is None, use current UTC year.
    """
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
    period: float = 1.0,
    base_year: Optional[int] = None,
    use_filename_time: bool = True,
    override_file_timestamps: bool = False,
    start_time: Optional[float] = None,
) -> OStream:
    """
    Load an O-stream file and (optionally) anchor its time axis to the filename stamp.

    Parameters
    ----------
    period : float
        Sampling period (seconds/sample) used to synthesize timestamps when needed
        (e.g., single-column CSVs). Default 1.0.
    base_year : int | None
        Year to use for filename-based timestamps (since the stamp has no year).
        If None, current UTC year is used. Usually pass the P-stream year.
    use_filename_time : bool
        If True, use the M..-D..-H..-M..-S..-U.xxx stamp from the filename as global start time
        when synthesizing timestamps (single-column CSVs) or when overriding.
    override_file_timestamps : bool
        If True and the CSV has a timestamp column, ignore it and synthesize absolute timestamps
        from the filename stamp + period.
    start_time : float | None
        Explicit epoch seconds start time. If provided, this overrides filename detection.

    Behavior by format
    ------------------
    .npz:     uses arrays inside file.
    .json:    uses keys inside file.
    .csv:
      - Headered:
          * If override_file_timestamps=True OR there is only a single column: synthesize
            timestamps using (filename stamp or start_time) + period.
          * Else: use the detected timestamp column as-is (absolute or relative, whichever it is).
      - Headerless:
          * If 1 column → single channel, synthesize timestamps with (filename stamp or start_time) + period.
          * If ≥2 columns → first is timestamps, others channels (no override unless override_file_timestamps=True).
    """
    path = Path(path)
    stem = path.stem

    # Resolve absolute start time if needed
    file_start: Optional[float] = None
    if start_time is not None:
        file_start = float(start_time)
    elif use_filename_time:
        file_start = _parse_start_from_filename(stem, base_year=base_year)

    # -------- NPZ --------
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        session_id = data["session_id"].item() if "session_id" in data else stem
        timestamps = np.asarray(data.get("timestamps", []), dtype=float)
        channels = np.asarray(data.get("channels", []), dtype=float)
        meta = {k: data[k] for k in data.files if k not in {"session_id", "timestamps", "channels"}}
        return OStream(session_id, timestamps, channels, meta)

    # -------- JSON / NDJSON / TXT (JSON) --------
    if path.suffix in {".json", ".ndjson", ".txt"}:
        with open(path, "r", encoding="utf8") as fh:
            obj = json.load(fh)
        session_id = obj.get("session_id", stem)
        timestamps = np.asarray(obj.get("timestamps", []), dtype=float)
        channels = np.asarray(obj.get("channels", []), dtype=float)
        meta = {k: v for k, v in obj.items() if k not in {"session_id", "timestamps", "channels"}}
        return OStream(session_id, timestamps, channels, meta)

    # -------- CSV --------
    if path.suffix == ".csv":
        with open(path, "r", encoding="utf8", newline="") as fh:
            # Try headered CSV
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                fns = _clean_fieldnames(reader.fieldnames)
                reader.fieldnames = fns

                # SINGLE-COLUMN (headered): treat as 1 channel, synthesize timestamps
                if len(fns) == 1:
                    col = fns[0]
                    vals = [float(row[col]) for row in reader if row.get(col, "").strip() != ""]
                    n = len(vals)
                    if file_start is None:
                        file_start = 0.0  # fallback
                    ts = file_start + period * np.arange(n, dtype=float)
                    ch = np.asarray(vals, dtype=float).reshape(n, 1)
                    return OStream(
                        session_id=stem,
                        timestamps=ts,
                        channels=ch,
                        meta={"columns": fns, "mode": "csv_headered_single_col",
                              "period": period, "start_time": file_start},
                    )

                # MULTI-COLUMN (headered)
                ts_col: Optional[str] = next((c for c in fns if c in _TS_ALIASES), None)
                if ts_col is None:
                    ts_col = fns[0]
                channel_fields = [c for c in fns if c not in {ts_col, "session_id"}]

                rows = [row for row in reader if any((v or "").strip() for v in row.values())]
                session_id = rows[0].get("session_id", stem) if rows else stem

                if override_file_timestamps:
                    # Ignore file timestamps; synthesize from filename/start_time
                    n = len(rows)
                    if file_start is None:
                        file_start = 0.0
                    ts = file_start + period * np.arange(n, dtype=float)
                    ch = np.asarray([[float(r[c]) for c in channel_fields] for r in rows], dtype=float)
                    return OStream(
                        session_id=session_id,
                        timestamps=ts,
                        channels=ch,
                        meta={"columns": fns, "channel_fields": channel_fields,
                              "mode": "csv_headered_multi_col_override",
                              "period": period, "start_time": file_start},
                    )
                else:
                    # Use timestamp column as-is
                    ts = np.asarray([float(r[ts_col]) for r in rows], dtype=float)
                    ch = np.asarray([[float(r[c]) for c in channel_fields] for r in rows], dtype=float)
                    return OStream(
                        session_id=session_id,
                        timestamps=ts,
                        channels=ch,
                        meta={"columns": fns, "channel_fields": channel_fields,
                              "mode": "csv_headered_multi_col"},
                    )

            # Headerless → numeric matrix
            fh.seek(0)
            raw_rows = [r for r in csv.reader(fh) if any(cell.strip() for cell in r)]
            data = np.asarray(raw_rows, dtype=float)

            if data.ndim != 2:
                raise ValueError("CSV must be 2D")

            # SINGLE-COLUMN (headerless): synthesize timestamps, one channel
            if data.shape[1] == 1:
                n = data.shape[0]
                if file_start is None:
                    file_start = 0.0
                ts = file_start + period * np.arange(n, dtype=float)
                ch = data.astype(float)  # (N, 1)
                return OStream(
                    session_id=stem,
                    timestamps=ts,
                    channels=ch,
                    meta={"columns": None, "mode": "csv_headerless_single_col",
                          "period": period, "start_time": file_start},
                )

            # ≥2 columns: first = timestamps (kept as-is), rest = channels
            ts = data[:, 0].astype(float)
            ch = data[:, 1:].astype(float)
            if override_file_timestamps:
                # Replace first column with synthesized absolute timestamps
                n = ts.shape[0]
                if file_start is None:
                    file_start = 0.0
                ts = file_start + period * np.arange(n, dtype=float)
                mode = "csv_headerless_multi_col_override"
            else:
                mode = "csv_headerless_multi_col"

            return OStream(
                session_id=stem,
                timestamps=ts,
                channels=ch,
                meta={"columns": None, "mode": mode,
                      "period": period if "override" in mode or "single" in mode else None,
                      "start_time": file_start if "override" in mode or "single" in mode else None},
            )

    raise ValueError(f"Unsupported O-stream file format: {path.suffix}")
