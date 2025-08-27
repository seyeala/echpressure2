"""Simple in-memory tables used by the project.

The real project persists information about oscillator signals and their
relationship to pressure measurements in a database.  For the exercises in
this kata we only need a *very* small subset of that functionality, therefore
the tables are materialised completely in memory using Python data
structures.

Three logical tables are modelled:

``Signals``
    Individual oscillator samples.  Besides the raw value the table also
    stores optional alignment errors and derivative bounds.

``OscFiles``
    Associates the logical key with the path of the file from which the
    sample was obtained.

``File2PressureMap``
    Provides the mapping between a sample and its corresponding pressure
    label.

Every table uses the composite key ``(sid, file_stamp, idx)`` as a primary
key.  Convenience functions are provided to export either a consolidated
"tall" representation of all tables or the normalised individual mapping
tables.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple

Key = Tuple[str, str, int]


@dataclass
class SignalRow:
    """Row representation for the ``Signals`` table."""

    sid: str
    file_stamp: str
    idx: int
    value: float
    alignment_error: Optional[float] = None
    deriv_lo: Optional[float] = None
    deriv_hi: Optional[float] = None


class Signals:
    """In-memory table storing oscillator samples."""

    def __init__(self) -> None:
        self._rows: Dict[Key, SignalRow] = {}

    def add(
        self,
        sid: str,
        file_stamp: str,
        idx: int,
        value: float,
        alignment_error: Optional[float] = None,
        deriv_lo: Optional[float] = None,
        deriv_hi: Optional[float] = None,
    ) -> None:
        """Insert a sample row.

        Raises
        ------
        KeyError
            If the composite key ``(sid, file_stamp, idx)`` already exists.
        """

        key = (sid, file_stamp, idx)
        if key in self._rows:
            raise KeyError(f"duplicate primary key: {key}")
        self._rows[key] = SignalRow(sid, file_stamp, idx, value, alignment_error, deriv_lo, deriv_hi)

    def to_records(self) -> List[Mapping[str, object]]:
        """Return the table contents as a list of dictionaries."""

        return [asdict(row) for row in self._rows.values()]

    def __iter__(self) -> Iterator[SignalRow]:
        return iter(self._rows.values())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def keys(self) -> Iterable[Key]:
        return self._rows.keys()

    def get(self, key: Key) -> Optional[SignalRow]:  # pragma: no cover - trivial
        return self._rows.get(key)


@dataclass
class OscFileRow:
    """Row representation for the ``OscFiles`` table."""

    sid: str
    file_stamp: str
    idx: int
    path: str


class OscFiles:
    """In-memory table mapping oscillator files to identifiers."""

    def __init__(self) -> None:
        self._rows: Dict[Key, OscFileRow] = {}

    def add(self, sid: str, file_stamp: str, idx: int, path: str) -> None:
        key = (sid, file_stamp, idx)
        if key in self._rows:
            raise KeyError(f"duplicate primary key: {key}")
        self._rows[key] = OscFileRow(sid, file_stamp, idx, path)

    def to_records(self) -> List[Mapping[str, object]]:
        return [asdict(row) for row in self._rows.values()]

    def __iter__(self) -> Iterator[OscFileRow]:
        return iter(self._rows.values())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def keys(self) -> Iterable[Key]:
        return self._rows.keys()

    def get(self, key: Key) -> Optional[OscFileRow]:  # pragma: no cover - trivial
        return self._rows.get(key)


@dataclass
class File2PressureRow:
    """Row representation for the ``File2PressureMap`` table."""

    sid: str
    file_stamp: str
    idx: int
    pressure_label: str


class File2PressureMap:
    """In-memory table mapping files to pressure labels."""

    def __init__(self) -> None:
        self._rows: Dict[Key, File2PressureRow] = {}

    def add(self, sid: str, file_stamp: str, idx: int, pressure_label: str) -> None:
        key = (sid, file_stamp, idx)
        if key in self._rows:
            raise KeyError(f"duplicate primary key: {key}")
        self._rows[key] = File2PressureRow(sid, file_stamp, idx, pressure_label)

    def to_records(self) -> List[Mapping[str, object]]:
        return [asdict(row) for row in self._rows.values()]

    def __iter__(self) -> Iterator[File2PressureRow]:
        return iter(self._rows.values())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def keys(self) -> Iterable[Key]:
        return self._rows.keys()

    def get(self, key: Key) -> Optional[File2PressureRow]:  # pragma: no cover - trivial
        return self._rows.get(key)


def export_tables(
    signals: Signals,
    osc_files: OscFiles,
    mappings: File2PressureMap,
    *,
    tall: bool = False,
) -> Mapping[str, object]:
    """Export stored data.

    Parameters
    ----------
    signals, osc_files, mappings:
        Table instances whose contents will be exported.
    tall:
        When ``True`` a single list of consolidated rows is returned.  When
        ``False`` (the default) a mapping of table name to rows is produced.
    """

    if tall:
        # Merge keys from all tables and produce one consolidated list of
        # records.  Sorting provides deterministic output which simplifies
        # testing and downstream processing.
        keys = set(signals.keys()) | set(osc_files.keys()) | set(mappings.keys())
        out: List[MutableMapping[str, object]] = []
        for key in sorted(keys):  # type: ignore[arg-type]
            row: Dict[str, object] = {"sid": key[0], "file_stamp": key[1], "idx": key[2]}
            if key in osc_files._rows:
                row["path"] = osc_files._rows[key].path
            if key in signals._rows:
                sig = signals._rows[key]
                row.update(
                    {
                        "value": sig.value,
                        "alignment_error": sig.alignment_error,
                        "deriv_lo": sig.deriv_lo,
                        "deriv_hi": sig.deriv_hi,
                    }
                )
            if key in mappings._rows:
                row["pressure_label"] = mappings._rows[key].pressure_label
            out.append(row)
        return out

    return {
        "signals": signals.to_records(),
        "osc_files": osc_files.to_records(),
        "file2pressure": mappings.to_records(),
    }


__all__ = [
    "SignalRow",
    "Signals",
    "OscFileRow",
    "OscFiles",
    "File2PressureRow",
    "File2PressureMap",
    "export_tables",
]

