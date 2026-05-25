from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from echopress.core.config_io import merge_config, write_resolved_config


@dataclass(frozen=True)
class AlignCleanerConfig:
    align_table: Path
    output_dir: Path
    config: Optional[Path] = None
    alignment_error_max: Optional[float] = 1.0
    pressure_min: Optional[float] = None
    pressure_max: Optional[float] = None


def _resolve_config(cfg: AlignCleanerConfig) -> dict[str, Any]:
    default_yml = Path(__file__).resolve().parents[3] / "configs" / "align_clean.default.yml"
    rcfg = merge_config(default_yaml_path=default_yml, user_yaml_path=cfg.config, cli_values=asdict(cfg))
    rcfg["align_table"] = str(cfg.align_table)
    rcfg["output_dir"] = str(cfg.output_dir)
    return rcfg


def run_align_clean(cfg: AlignCleanerConfig) -> dict[str, Any]:
    rcfg = _resolve_config(cfg)
    out_dir = Path(rcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out_dir / "clean-align_config.resolved.yml")

    rows = pd.read_json(Path(rcfg["align_table"]))
    rows = rows.dropna(subset=["path", "pressure_value"]).copy()
    if "alignment_error" in rows.columns and rcfg.get("alignment_error_max") is not None:
        rows["alignment_error"] = pd.to_numeric(rows["alignment_error"], errors="coerce")
        rows = rows[rows["alignment_error"].notna() & (rows["alignment_error"] <= float(rcfg["alignment_error_max"]))]
    if rcfg.get("pressure_min") is not None:
        rows = rows[rows["pressure_value"] >= float(rcfg["pressure_min"])]
    if rcfg.get("pressure_max") is not None:
        rows = rows[rows["pressure_value"] <= float(rcfg["pressure_max"])]
    rows = rows.drop_duplicates(subset=["path"]).reset_index(drop=True)
    clean_path = out_dir / "align.cleaned.json"
    clean_path.write_text(rows.to_json(orient="records", indent=2), encoding="utf-8")
    summary = {"input_rows": int(len(pd.read_json(Path(rcfg['align_table'])))), "output_rows": int(len(rows)), "output": str(clean_path)}
    (out_dir / "clean_align_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
