from __future__ import annotations

"""Configuration utilities for echopress.

This module defines a hierarchical configuration schema using dataclasses.
The :class:`Settings` container groups several specialised sub-sections such
as calibration coefficients and quality controls.  Instances may be populated
from environment variables or from YAML/JSON files with matching nested
keys.
"""

from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Settings schema
# ---------------------------------------------------------------------------


@dataclass
class CalibrationSettings:
    """Per-channel affine calibration coefficients."""

    alpha: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    beta: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MappingSettings:
    """Parameters controlling stream alignment and derivative estimates."""

    tie_breaker: str = "earliest"
    O_max: float = 0.1
    W: int = 5
    kappa: float = 1.0


@dataclass
class PressureSettings:
    """Options related to scalar pressure extraction."""

    scalar_channel: int = 2


@dataclass
class UnitsSettings:
    """Physical units for common quantities."""

    pressure: str = "Pa"
    voltage: str = "V"


@dataclass
class TimestampSettings:
    """Controls for parsing timestamp fields."""

    format: str | None = None
    timezone: str = "UTC"
    year_fallback: int = field(
        default_factory=lambda: datetime.now(timezone.utc).year
    )


@dataclass
class QualitySettings:
    """Quality control toggles."""

    reject_if_Ealign_gt_Omax: bool = True
    min_records_in_W: int = 3


@dataclass
class IngestSettings:
    """Options controlling dataset ingestion."""

    pstream_csv_patterns: list[str] = field(
        default_factory=lambda: ["voltprsr"]
    )


@dataclass
class Settings:
    """Container for all runtime configuration sections."""

    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    mapping: MappingSettings = field(default_factory=MappingSettings)
    pressure: PressureSettings = field(default_factory=PressureSettings)
    units: UnitsSettings = field(default_factory=UnitsSettings)
    timestamp: TimestampSettings = field(default_factory=TimestampSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from ``ECHOPRESS_*`` environment variables."""

        settings = cls()

        def set_path(path: str, value: Any) -> None:
            obj: Any = settings
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        mapping = {
            "calibration.alpha": ("ECHOPRESS_CALIBRATION_ALPHA", lambda v: [float(x) for x in v.split(",")]),
            "calibration.beta": ("ECHOPRESS_CALIBRATION_BETA", lambda v: [float(x) for x in v.split(",")]),
            "mapping.tie_breaker": ("ECHOPRESS_TIE_BREAKER", str),
            "mapping.O_max": ("ECHOPRESS_O_MAX", float),
            "mapping.W": ("ECHOPRESS_W", int),
            "mapping.kappa": ("ECHOPRESS_KAPPA", float),
            "pressure.scalar_channel": ("ECHOPRESS_PRESSURE_SCALAR_CHANNEL", int),
            "quality.reject_if_Ealign_gt_Omax": (
                "ECHOPRESS_REJECT_IF_EALIGN_GT_OMAX",
                lambda v: v.lower() in {"1", "true", "yes"},
            ),
            "quality.min_records_in_W": ("ECHOPRESS_MIN_RECORDS_IN_W", int),
            "units.pressure": ("ECHOPRESS_UNITS_PRESSURE", str),
            "units.voltage": ("ECHOPRESS_UNITS_VOLTAGE", str),
            "timestamp.format": ("ECHOPRESS_TIMESTAMP_FORMAT", str),
            "timestamp.timezone": ("ECHOPRESS_TIMESTAMP_TIMEZONE", str),
            "timestamp.year_fallback": ("ECHOPRESS_TIMESTAMP_YEAR_FALLBACK", int),
            "ingest.pstream_csv_patterns": (
                "ECHOPRESS_INGEST_PSTREAM_CSV_PATTERNS",
                lambda v: [s.strip() for s in v.split(",") if s.strip()],
            ),
        }
        for path, (env, conv) in mapping.items():
            if env in os.environ:
                set_path(path, conv(os.environ[env]))
        return settings


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


def _update_from_dict(obj: Any, data: Dict[str, Any]) -> None:
    """Recursively update dataclass ``obj`` with ``data``."""

    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        attr = getattr(obj, key)
        if is_dataclass(attr):
            _update_from_dict(attr, value)
        else:
            setattr(obj, key, value)


def load_settings(path: str | Path) -> Settings:
    """Load settings from a JSON or YAML file."""

    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML files")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    settings = Settings()
    if isinstance(data, dict):
        _update_from_dict(settings, data)
    return settings
