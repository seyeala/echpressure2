"""Parser for O-stream files.

O-streams typically contain multi-channel observation arrays, per-sample
timestamps and metadata such as session identifiers.  This module provides
:func:`load_ostream` to load these files into the :class:`OStream` data class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json

import numpy as np


@dataclass
class OStream:
    """In-memory representation of an O-stream."""

    session_id: str
    timestamps: np.ndarray
    channels: np.ndarray
    meta: Dict[str, Any]


def load_ostream(path: str | Path) -> OStream:
    """Load an O-stream file.

    The loader understands two simple file formats:

    - ``.npz`` archives containing ``session_id``, ``timestamps`` and
      ``channels`` arrays, plus optional additional metadata arrays.
    - JSON files with the same keys.

    Additional arrays or key/value pairs are preserved in the ``meta``
    dictionary of the returned :class:`OStream` instance.
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        session_id = (
            data["session_id"].item() if "session_id" in data else path.stem
        )
        timestamps = np.asarray(data.get("timestamps", []), dtype=float)
        channels = np.asarray(data.get("channels", []), dtype=float)
        meta = {
            key: data[key]
            for key in data.files
            if key not in {"session_id", "timestamps", "channels"}
        }
        return OStream(session_id=session_id, timestamps=timestamps, channels=channels, meta=meta)

    if path.suffix in {".json", ".ndjson", ".txt"}:
        with open(path, "r", encoding="utf8") as fh:
            obj = json.load(fh)
        session_id = obj.get("session_id", path.stem)
        timestamps = np.asarray(obj.get("timestamps", []), dtype=float)
        channels = np.asarray(obj.get("channels", []), dtype=float)
        meta = {
            key: value
            for key, value in obj.items()
            if key not in {"session_id", "timestamps", "channels"}
        }
        return OStream(session_id=session_id, timestamps=timestamps, channels=channels, meta=meta)

    raise ValueError(f"Unsupported O-stream file format: {path.suffix}")
