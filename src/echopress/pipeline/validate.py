from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REQUIRED_ALIGN_COLS = ["path", "pressure_value"]


def validate_align_json(path: Path) -> Tuple[bool, Dict[str, object]]:
    if not path.exists():
        return False, {"reason": "missing"}
    try:
        df = pd.read_json(path)
    except Exception as exc:
        return False, {"reason": f"json_read_error: {exc}"}
    cols = list(df.columns)
    ok = all(c in cols for c in REQUIRED_ALIGN_COLS) and len(df) > 0
    return ok, {"row_count": int(len(df)), "actual_columns": cols, "required_columns": REQUIRED_ALIGN_COLS}


def validate_index_json(path: Path) -> Tuple[bool, Dict[str, object]]:
    if not path.exists():
        return False, {"reason": "missing"}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return False, {"reason": f"json_read_error: {exc}"}
    ok = isinstance(data, dict) and "pstreams" in data and "ostreams" in data
    return ok, {"keys": list(data.keys()) if isinstance(data, dict) else []}


def count_npz(dataset_root: Path) -> int:
    return sum(1 for _ in dataset_root.rglob("*.npz"))
