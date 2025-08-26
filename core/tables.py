"""In-memory tables for oscillator signals and pressure mapping.

This module provides simple container classes for working with oscillator
signal data.  Three logical tables are represented:

``Signals``
    Individual signal samples and associated metadata including alignment
    errors and derivative bounds.
``OscFiles``
    Mapping from oscillator data files to the logical key.
``File2PressureMap``
    Mapping between file/sample identifiers and pressure labels.

The tables are stored entirely in memory using dictionaries keyed by the
composite primary key ``(sid, file_stamp, idx)``.  Each table exposes a
minimal API for inserting rows and exporting the stored data either in a
"tall" consolidated form or as individual mapping tables.
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

    def keys(self) -> Iterable[Key]:
        return self._rows.keys()


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

    def keys(self) -> Iterable[Key]:
        return self._rows.keys()


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

    def keys(self) -> Iterable[Key]:
        return self._rows.keys()


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
        keys = set(signals.keys()) | set(osc_files.keys()) | set(mappings.keys())
        out: List[MutableMapping[str, object]] = []
        for key in keys:
            row: Dict[str, object] = {"sid": key[0], "file_stamp": key[1], "idx": key[2]}
            if key in osc_files._rows:
                row.update({"path": osc_files._rows[key].path})
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
                row.update({"pressure_label": mappings._rows[key].pressure_label})
            out.append(row)
        return out

    return {
        "signals": signals.to_records(),
        "osc_files": osc_files.to_records(),
        "file2pressure": mappings.to_records(),
    }
