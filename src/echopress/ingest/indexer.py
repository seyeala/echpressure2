"""Dataset indexer utilities.

The :class:`DatasetIndexer` class walks a dataset directory and builds
registries of available P-stream and O-stream files.  The indexer is
agnostic to the exact dataset layout; it simply records files matching
expected suffixes and exposes lookup helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

from ..config import Settings

PSTREAM_EXTENSIONS = {".pstream", ".p", ".ps"}
OSTREAM_EXTENSIONS = {".ostream", ".o", ".os", ".npz", ".json", ".csv"}


def _session_id(path: Path, patterns: Iterable[str] = ()) -> str:
    """Derive a session identifier from ``path``.

    The filename stem is compared against ``patterns`` in a case-insensitive
    manner.  Any matching prefix is stripped while preserving the original
    casing of the remaining characters.
    """

    stem = path.stem
    stem_lower = stem.lower()
    for prefix in sorted((p.lower() for p in patterns), key=len, reverse=True):
        if stem_lower.startswith(prefix):
            suffix = stem[len(prefix) :]
            return suffix or stem
    return stem


def _is_pstream_csv(path: Path, patterns: Iterable[str]) -> bool:
    """Return ``True`` if ``path`` matches a configured P-stream CSV pattern."""
    if path.suffix.lower() != ".csv":
        return False
    stem = path.stem
    stem_lower = stem.lower()
    for pattern in patterns:
        try:
            if re.match(pattern, stem, flags=re.IGNORECASE):
                return True
        except re.error:
            if stem_lower.startswith(pattern.lower()):
                return True
    return False


@dataclass
class DatasetIndexer:
    """Index of dataset files on disk."""

    root: Path
    pstreams: Dict[str, List[Path]] = field(default_factory=dict)
    ostreams: Dict[str, List[Path]] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)
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
        self._pstream_keys.clear()
        self._ostream_keys.clear()

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if _is_pstream_csv(path, self.settings.ingest.pstream_csv_patterns):
                sid = path.stem
                self.pstreams.setdefault(sid, []).append(path)
                self._pstream_keys.setdefault(sid.lower(), sid)
                continue
            sid = _session_id(path)
            suffix = path.suffix.lower()
            if suffix in PSTREAM_EXTENSIONS:
                self.pstreams.setdefault(sid, []).append(path)
                self._pstream_keys.setdefault(sid.lower(), sid)
            elif suffix in OSTREAM_EXTENSIONS:
                self.ostreams.setdefault(sid, []).append(path)
                self._ostream_keys.setdefault(sid.lower(), sid)

    # ------------------------------------------------------------------
    def sessions(self) -> List[str]:
        """Return the session identifiers known to the indexer."""
        return sorted(set(self.pstreams) | set(self.ostreams))

    def get_pstreams(self, session_id: str) -> List[Path]:
        """Return P-stream files for ``session_id`` (possibly empty)."""
        key = self._pstream_keys.get(session_id.lower())
        return self.pstreams.get(key, []) if key is not None else []

    def get_ostreams(self, session_id: str) -> List[Path]:
        """Return O-stream files for ``session_id`` (possibly empty)."""
        key = self._ostream_keys.get(session_id.lower())
        return self.ostreams.get(key, []) if key is not None else []

    def first_pstream(self, session_id: str) -> Optional[Path]:
        """Convenience method returning the first P-stream file for a session."""
        files = self.get_pstreams(session_id)
        return files[0] if files else None

    def first_ostream(self, session_id: str) -> Optional[Path]:
        """Convenience method returning the first O-stream file for a session."""
        files = self.get_ostreams(session_id)
        return files[0] if files else None

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"DatasetIndexer(root={self.root!r}, "
            f"pstreams={len(self.pstreams)}, ostreams={len(self.ostreams)})"
        )
