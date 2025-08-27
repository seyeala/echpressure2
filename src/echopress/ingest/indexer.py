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

from ..config import Settings

PSTREAM_EXTENSIONS = {".pstream", ".p", ".ps"}
OSTREAM_EXTENSIONS = {".ostream", ".o", ".os", ".npz", ".json", ".csv"}


def _session_id(path: Path) -> str:
    """Derive a session identifier from ``path``.

    The default strategy uses the stem (basename without extensions).
    """
    return path.stem


def _is_pstream_csv(path: Path, patterns: Iterable[str]) -> bool:
    """Return ``True`` if ``path`` matches a configured P-stream CSV pattern."""
    if path.suffix.lower() != ".csv":
        return False
    stem = path.stem.lower()
    return any(p.lower() in stem for p in patterns)


@dataclass
class DatasetIndexer:
    """Index of dataset files on disk."""

    root: Path
    pstreams: Dict[str, List[Path]] = field(default_factory=dict)
    ostreams: Dict[str, List[Path]] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.scan()

    # ------------------------------------------------------------------
    def scan(self) -> None:
        """Walk the dataset tree and populate the registries."""
        self.pstreams.clear()
        self.ostreams.clear()

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            sid = _session_id(path)
            if _is_pstream_csv(path, self.settings.pstream_csv_patterns):
                self.pstreams.setdefault(sid, []).append(path)
                continue
            suffix = path.suffix.lower()
            if suffix in PSTREAM_EXTENSIONS:
                self.pstreams.setdefault(sid, []).append(path)
            elif suffix in OSTREAM_EXTENSIONS:
                self.ostreams.setdefault(sid, []).append(path)

    # ------------------------------------------------------------------
    def sessions(self) -> List[str]:
        """Return the session identifiers known to the indexer."""
        return sorted(set(self.pstreams) | set(self.ostreams))

    def get_pstreams(self, session_id: str) -> List[Path]:
        """Return P-stream files for ``session_id`` (possibly empty)."""
        return self.pstreams.get(session_id, [])

    def get_ostreams(self, session_id: str) -> List[Path]:
        """Return O-stream files for ``session_id`` (possibly empty)."""
        return self.ostreams.get(session_id, [])

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
