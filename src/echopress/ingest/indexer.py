# src/echopress/ingest/indexer.py
"""Dataset indexer utilities (window-mode friendly).

Builds registries of P-stream and O-stream files. Sessions are the full
filename stem (no stripping) to avoid ambiguity. Supports classifying
CSV pressure files by name pattern (e.g., 'voltprsr*.csv').

Also provides:
- flat accessors (all_pstreams / all_ostreams)
- fallback in get_*streams(..., fallback=True) to project-wide lists
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

from ..config import Settings

# Extension-based classification
PSTREAM_EXTENSIONS = {".pstream", ".p", ".ps"}
OSTREAM_EXTENSIONS = {".ostream", ".o", ".os", ".npz", ".json", ".csv"}


def _session_id(path: Path) -> str:
    """Session is the full filename stem (no stripping)."""
    return path.stem


def _is_pstream_csv(path: Path, patterns: Iterable[str] | None) -> bool:
    """Return True if '.csv' filename matches any configured P-stream pattern."""
    if path.suffix.lower() != ".csv":
        return False
    if not patterns:
        # Sensible defaults for this repo
        patterns = ("voltprsr", "ai_log")
    stem = path.stem
    stem_lower = stem.lower()
    for pattern in patterns:
        try:
            if re.match(pattern, stem, flags=re.IGNORECASE):
                return True
        except re.error:
            if stem_lower.startswith(pattern.lower()) or pattern.lower() in stem_lower:
                return True
    return False


@dataclass
class DatasetIndexer:
    """Index of dataset files on disk."""
    root: Path
    # Per-session registries
    pstreams: Dict[str, List[Path]] = field(default_factory=dict)
    ostreams: Dict[str, List[Path]] = field(default_factory=dict)
    # Flat lists regardless of session
    p_all: List[Path] = field(default_factory=list)
    o_all: List[Path] = field(default_factory=list)
    # Settings (for patterns)
    settings: Settings = field(default_factory=Settings)
    # Case-insensitive maps for session lookup
    _pstream_keys: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _ostream_keys: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.scan()

    def scan(self) -> None:
        """Walk the dataset tree and populate the registries."""
        self.pstreams.clear(); self.ostreams.clear()
        self.p_all.clear(); self.o_all.clear()
        self._pstream_keys.clear(); self._ostream_keys.clear()

        ingest_cfg = getattr(self.settings, "ingest", None)
        pstream_csv_patterns = tuple(getattr(ingest_cfg, "pstream_csv_patterns", ()))

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            sid = _session_id(path)
            suffix = path.suffix.lower()

            if _is_pstream_csv(path, pstream_csv_patterns):
                self.pstreams.setdefault(sid, []).append(path)
                self.p_all.append(path)
                self._pstream_keys.setdefault(sid.lower(), sid)
                continue

            if suffix in PSTREAM_EXTENSIONS:
                self.pstreams.setdefault(sid, []).append(path)
                self.p_all.append(path)
                self._pstream_keys.setdefault(sid.lower(), sid)
            elif suffix in OSTREAM_EXTENSIONS:
                self.ostreams.setdefault(sid, []).append(path)
                self.o_all.append(path)
                self._ostream_keys.setdefault(sid.lower(), sid)
            # else: ignore

    # Sessions
    def sessions(self) -> List[str]:
        return sorted(set(self.pstreams) | set(self.ostreams))

    # Lookups (with optional fallback to project-wide lists)
    def get_pstreams(self, session_id: str, fallback: bool = True) -> List[Path]:
        key = self._pstream_keys.get(session_id.lower(), session_id)
        files = self.pstreams.get(key, [])
        return files if files or not fallback else list(self.p_all)

    def get_ostreams(self, session_id: str, fallback: bool = True) -> List[Path]:
        key = self._ostream_keys.get(session_id.lower(), session_id)
        files = self.ostreams.get(key, [])
        return files if files or not fallback else list(self.o_all)

    def first_pstream(self, session_id: str, fallback: bool = True) -> Optional[Path]:
        files = self.get_pstreams(session_id, fallback=fallback)
        return files[0] if files else None

    def first_ostream(self, session_id: str, fallback: bool = True) -> Optional[Path]:
        files = self.get_ostreams(session_id, fallback=fallback)
        return files[0] if files else None

    # Flat accessors
    def all_pstreams(self) -> List[Path]:
        return list(self.p_all)

    def all_ostreams(self) -> List[Path]:
        return list(self.o_all)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DatasetIndexer(root={self.root!r}, pstreams={len(self.p_all)}, ostreams={len(self.o_all)})"
