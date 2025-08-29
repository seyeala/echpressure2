"""Dataset indexer utilities.

The :class:`DatasetIndexer` class walks a dataset directory and builds
registries of available P-stream and O-stream files. The indexer records files
matching expected suffixes and/or name patterns, exposes lookup helpers, and
also provides flat lists regardless of session ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

from ..config import Settings

# Extension-based classification (kept for backward compatibility)
PSTREAM_EXTENSIONS = {".pstream", ".p", ".ps"}
OSTREAM_EXTENSIONS = {".ostream", ".o", ".os", ".npz", ".json", ".csv"}


def _session_id(path: Path, patterns: Iterable[str] = ()) -> str:
    """Derive a session identifier from ``path``.

    If ``patterns`` (prefixes) are provided, strip the first matching prefix
    (case-insensitive) from the filename stem; otherwise, use the stem as-is.
    """
    stem = path.stem
    stem_lower = stem.lower()
    # Try longer prefixes first to avoid partial matches winning
    for prefix in sorted((p.lower() for p in (patterns or ())), key=len, reverse=True):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix):]
            return suffix or stem
    return stem


def _is_pstream_csv(path: Path, patterns: Iterable[str]) -> bool:
    """Return True if a .csv file name matches any configured P-stream pattern.

    Patterns may be regular expressions; if a pattern is not a valid regex,
    it is treated as a case-insensitive 'starts with' prefix.
    """
    if path.suffix.lower() != ".csv":
        return False
    stem = path.stem
    stem_lower = stem.lower()
    for pattern in (patterns or ()):
        try:
            if re.match(pattern, stem, flags=re.IGNORECASE):
                return True
        except re.error:
            # Not a valid regex; treat as prefix
            if stem_lower.startswith(pattern.lower()):
                return True
    return False


@dataclass
class DatasetIndexer:
    """Index of dataset files on disk."""

    root: Path
    # Per-session registries
    pstreams: Dict[str, List[Path]] = field(default_factory=dict)
    ostreams: Dict[str, List[Path]] = field(default_factory=dict)

    # Flat lists regardless of session (easy access)
    p_all: List[Path] = field(default_factory=list)
    o_all: List[Path] = field(default_factory=list)

    # Settings (optionally provides ingest config)
    settings: Settings = field(default_factory=Settings)

    # Case-insensitive maps for session lookup -> canonical session ID
    _pstream_keys: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _ostream_keys: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.scan()

    # ------------------------------------------------------------------
    def scan(self) -> None:
        """Walk the dataset tree and populate the registries."""
        self.pstreams.clear()
        self.ostreams.clear()
        self.p_all.clear()
        self.o_all.clear()
        self._pstream_keys.clear()
        self._ostream_keys.clear()

        ingest_cfg = getattr(self.settings, "ingest", None)
        pstream_csv_patterns = tuple(getattr(ingest_cfg, "pstream_csv_patterns", ()))
        session_prefixes = tuple(getattr(ingest_cfg, "session_prefixes", ()))

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            # 1) CSV-by-name rule (lets you keep both streams in .csv)
            if _is_pstream_csv(path, pstream_csv_patterns):
                sid = _session_id(path, session_prefixes)
                self.pstreams.setdefault(sid, []).append(path)
                self.p_all.append(path)
                self._pstream_keys.setdefault(sid.lower(), sid)
                continue

            # 2) Extension-based fallback
            suffix = path.suffix.lower()
            if suffix in PSTREAM_EXTENSIONS:
                sid = _session_id(path, session_prefixes)
                self.pstreams.setdefault(sid, []).append(path)
                self.p_all.append(path)
                self._pstream_keys.setdefault(sid.lower(), sid)
            elif suffix in OSTREAM_EXTENSIONS:
                sid = _session_id(path, session_prefixes)
                self.ostreams.setdefault(sid, []).append(path)
                self.o_all.append(path)
                self._ostream_keys.setdefault(sid.lower(), sid)
            # else: ignore unknown files

    # ------------------------------------------------------------------
    # Session utilities
    def sessions(self) -> List[str]:
        """Return the session identifiers known to the indexer."""
        return sorted(set(self.pstreams) | set(self.ostreams))

    def get_pstreams(self, session_id: str, fallback: bool = True) -> List[Path]:
        """Return P-stream files for ``session_id``.
        If none found and ``fallback`` is True, return all project P-streams."""
        key = self._pstream_keys.get(session_id.lower(), session_id)
        files = self.pstreams.get(key, [])
        if not files and fallback:
            return list(self.p_all)
        return files

    def get_ostreams(self, session_id: str, fallback: bool = True) -> List[Path]:
        """Return O-stream files for ``session_id``.
        If none found and ``fallback`` is True, return all project O-streams."""
        key = self._ostream_keys.get(session_id.lower(), session_id)
        files = self.ostreams.get(key, [])
        if not files and fallback:
            return list(self.o_all)
        return files

    def first_pstream(self, session_id: str, fallback: bool = True) -> Optional[Path]:
        """Return the first P-stream for a session (or first project P-stream if falling back)."""
        files = self.get_pstreams(session_id, fallback=fallback)
        return files[0] if files else None

    def first_ostream(self, session_id: str, fallback: bool = True) -> Optional[Path]:
        """Return the first O-stream for a session (or first project O-stream if falling back)."""
        files = self.get_ostreams(session_id, fallback=fallback)
        return files[0] if files else None

    # ------------------------------------------------------------------
    # Flat accessors (regardless of session)
    def all_pstreams(self) -> List[Path]:
        """Return all P-stream files discovered (across all sessions)."""
        return list(self.p_all)

    def all_ostreams(self) -> List[Path]:
        """Return all O-stream files discovered (across all sessions)."""
        return list(self.o_all)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"DatasetIndexer(root={self.root!r}, "
            f"pstreams={len(self.p_all)}, ostreams={len(self.o_all)})"
        )
