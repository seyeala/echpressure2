from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_alignment_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows = json.loads(p.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Alignment table must be a JSON list: {p}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Alignment table rows must be JSON objects: {p}")
    return rows


def save_alignment_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2, default=float))
    return p


def row_key(row: dict[str, Any], match_key: str, row_index: int | None = None) -> str:
    """
    Supported match keys:
      - path
      - path_basename
      - file_stamp
      - sid
      - sid_file_stamp
      - row_index
    """
    if match_key == "row_index":
        if row_index is None:
            raise ValueError("row_index matching requires row_index")
        return str(row_index)

    if match_key == "path":
        return str(row.get("path", ""))

    if match_key == "path_basename":
        return Path(str(row.get("path", ""))).name

    if match_key == "file_stamp":
        return str(row.get("file_stamp", ""))

    if match_key == "sid":
        return str(row.get("sid", ""))

    if match_key == "sid_file_stamp":
        return f"{row.get('sid', '')}::{row.get('file_stamp', '')}"

    raise ValueError(f"Unsupported match_key: {match_key}")


def _item_key(item: Any, match_key: str) -> str:
    """
    Removal list can contain:
      - strings
      - integers for row_index
      - objects with path/file_stamp/sid fields
    """
    if isinstance(item, (str, int, float)):
        if match_key == "path_basename":
            return Path(str(item)).name
        return str(int(item)) if match_key == "row_index" else str(item)

    if not isinstance(item, dict):
        raise ValueError(f"Unsupported removal-list item: {item!r}")

    if match_key == "sid_file_stamp":
        return f"{item.get('sid', '')}::{item.get('file_stamp', '')}"

    if match_key == "path_basename":
        return Path(str(item.get("path", item.get("resolved_path", "")))).name

    if match_key == "row_index":
        return str(item.get("row_index", item.get("index", "")))

    return str(item.get(match_key, item.get("path", "")))


def load_remove_keys(remove_list: str | Path, match_key: str) -> set[str]:
    """
    Supports:
      - JSON list of strings or objects
      - TXT one item per line
      - CSV with columns like path, file_stamp, sid, row_index
    """
    p = Path(remove_list)
    suffix = p.suffix.lower()

    if suffix == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError("Removal JSON must be a list")
        return {_item_key(item, match_key) for item in data}

    if suffix in {".txt", ".list"}:
        return {
            Path(line.strip()).name if match_key == "path_basename" else line.strip()
            for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    if suffix == ".csv":
        keys: set[str] = set()
        with p.open("r", newline="", encoding="utf8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                keys.add(_item_key(row, match_key))
        return keys

    raise ValueError(f"Unsupported remove-list format: {p.suffix}")


def revise_alignment_by_remove_list(
    *,
    align_table: str | Path,
    remove_list: str | Path,
    output: str | Path,
    match_key: str = "path",
    invert: bool = False,
) -> dict[str, Any]:
    rows = load_alignment_rows(align_table)
    remove_keys = load_remove_keys(remove_list, match_key)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        key = row_key(row, match_key, row_index=idx)
        should_remove = key in remove_keys

        if invert:
            should_remove = not should_remove

        if should_remove:
            removed.append(row)
        else:
            kept.append(row)

    save_alignment_rows(kept, output)

    return {
        "input": str(align_table),
        "remove_list": str(remove_list),
        "output": str(output),
        "match_key": match_key,
        "input_rows": len(rows),
        "remove_key_count": len(remove_keys),
        "removed_rows": len(removed),
        "kept_rows": len(kept),
    }
