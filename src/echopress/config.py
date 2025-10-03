from __future__ import annotations

"""Configuration utilities for echopress.

This module defines a hierarchical configuration schema using Pydantic models.
The :class:`Settings` container groups several specialised sub-sections such as
calibration coefficients, quality controls, dataset metadata, and adapter
options.  Instances can be populated from environment variables or from
YAML/JSON files with matching nested keys.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _split_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class SectionModel(BaseModel):
    """Base model for configuration subsections that ignores unknown fields."""

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Settings schema
# ---------------------------------------------------------------------------


class CalibrationSettings(SectionModel):
    """Per-channel affine calibration coefficients."""

    alpha: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    beta: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @field_validator("alpha", "beta", mode="before")
    @classmethod
    def _coerce_float_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _split_floats(value)
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return value


class MappingSettings(SectionModel):
    """Parameters controlling stream alignment and derivative estimates."""

    tie_breaker: str = "earliest"
    O_max: float = 0.0001
    W: int = 5
    kappa: float = 3.0


class PressureSettings(SectionModel):
    """Options related to scalar pressure extraction."""

    scalar_channel: int = 2


class UnitsSettings(SectionModel):
    """Physical units for common quantities."""

    pressure: str = "Pa"
    voltage: str = "V"


class TimestampSettings(SectionModel):
    """Controls for parsing timestamp fields."""

    format: str | None = None
    timezone: str = "UTC"
    year_fallback: int = 1970


class QualitySettings(SectionModel):
    """Quality control toggles."""

    reject_if_Ealign_gt_Omax: bool = True
    min_records_in_W: int = 3


class IngestSettings(SectionModel):
    """Options controlling dataset ingestion."""

    pstream_csv_patterns: list[str] = Field(default_factory=lambda: ["voltprsr"])

    @field_validator("pstream_csv_patterns", mode="before")
    @classmethod
    def _coerce_patterns(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _split_strings(value)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return value


class DatasetSettings(SectionModel):
    """Metadata about input datasets."""

    root: str = "."
    timezone: str = "UTC"
    timestamp_grammar: str = (
        "ISO-8601, HH:MM:SS[.ffffff], floating-point seconds since the Unix epoch,\n"
        "or Mxx-Dxx-Hxx-Mxx-Sxx-U.xxx"
    )
    ostream: str | None = None
    pstream: str | None = None


class AlignSettings(SectionModel):
    """Defaults for stream alignment CLI helpers."""

    duration: float = 0.02
    window_mode: bool = True
    base_year: int | None = None


class PeriodEstimateSettings(SectionModel):
    """Sub-config describing spectral period estimation."""

    fs: float = 1.0
    f0: float = 1.0


class AdapterSettings(SectionModel):
    """Configuration for adapter execution."""

    name: str = "cec"
    output_length: int = 0
    period_est: PeriodEstimateSettings = Field(default_factory=PeriodEstimateSettings)
    pr_min: float | None = None
    pr_max: float | None = None
    n: int = 1
    plot: bool = False
    align_table: str = Field(default_factory=lambda: str(Path(DatasetSettings().root) / "align.json"))
    seed: int = 0


class VizSettings(SectionModel):
    """Configuration for simple visualisation helpers."""

    title: str = "Signal"
    save: str | None = None


class Settings(BaseSettings):
    """Container for all runtime configuration sections."""

    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    mapping: MappingSettings = Field(default_factory=MappingSettings)
    pressure: PressureSettings = Field(default_factory=PressureSettings)
    units: UnitsSettings = Field(default_factory=UnitsSettings)
    timestamp: TimestampSettings = Field(default_factory=TimestampSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    align: AlignSettings = Field(default_factory=AlignSettings)
    adapter: AdapterSettings = Field(default_factory=AdapterSettings)
    viz: VizSettings = Field(default_factory=VizSettings)

    model_config = SettingsConfigDict(
        env_prefix="ECHOPRESS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        class LegacyEnvSettingsSource(EnvSettingsSource):
            def decode_complex_value(self, field_name, target_field, value):  # type: ignore[override]
                try:
                    return super().decode_complex_value(field_name, target_field, value)
                except json.JSONDecodeError:
                    return value

        env_settings.__class__ = LegacyEnvSettingsSource
        return init_settings, env_settings, dotenv_settings, file_secret_settings


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


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
    if not isinstance(data, dict):
        raise TypeError("Configuration file must define a mapping")
    return Settings.model_validate(data)
