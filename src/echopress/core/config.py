"""Configuration helpers for the echopress core modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - optional dependency guard
    import yaml
except ImportError as exc:  # pragma: no cover - runtime error
    raise RuntimeError("PyYAML is required to load configuration") from exc


@dataclass
class CalibrationConfig:
    """Affine calibration coefficients."""

    alpha: Sequence[float]
    beta: Sequence[float]


@dataclass
class PressureConfig:
    """Pressure channel selection."""

    scalar_channel: int


@dataclass
class MappingConfig:
    """Parameters controlling O-stream to P-stream mapping."""

    O_max: float
    tie_breaker: str = "earliest"


@dataclass
class DerivativeConfig:
    """Derivative estimation parameters."""

    W: int


@dataclass
class UncertaintyConfig:
    """Uncertainty model parameters."""

    kappa: float


@dataclass
class DatasetConfig:
    """Aggregate configuration object."""

    calibration: CalibrationConfig
    pressure: PressureConfig
    mapping: MappingConfig
    derivative: DerivativeConfig
    uncertainty: UncertaintyConfig


def load_config(path: str | Path) -> DatasetConfig:
    """Load a :class:`DatasetConfig` from a YAML file."""

    with open(path, "r", encoding="utf8") as fh:
        data = yaml.safe_load(fh)

    calib = data.get("calibration", {})
    press = data.get("pressure", {})
    mapping = data.get("mapping", {})
    deriv = data.get("derivative", {})
    uncert = data.get("uncertainty", {})

    calib_cfg = CalibrationConfig(
        alpha=list(calib.get("alpha", [])),
        beta=list(calib.get("beta", [])),
    )
    pressure_cfg = PressureConfig(
        scalar_channel=int(press.get("scalar_channel", 0)),
    )
    mapping_cfg = MappingConfig(
        O_max=float(mapping.get("O_max", float("inf"))),
        tie_breaker=str(mapping.get("tie_breaker", "earliest")),
    )
    derivative_cfg = DerivativeConfig(
        W=int(deriv.get("W", 1)),
    )
    uncertainty_cfg = UncertaintyConfig(
        kappa=float(uncert.get("kappa", 1.0)),
    )

    return DatasetConfig(
        calibration=calib_cfg,
        pressure=pressure_cfg,
        mapping=mapping_cfg,
        derivative=derivative_cfg,
        uncertainty=uncertainty_cfg,
    )


__all__ = [
    "CalibrationConfig",
    "PressureConfig",
    "MappingConfig",
    "DerivativeConfig",
    "UncertaintyConfig",
    "DatasetConfig",
    "load_config",
]
