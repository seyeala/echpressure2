"""Configuration handling for the :mod:`echopress` package.

The project makes heavy use of tunable parameters for calibration and
mapping.  This module provides a small dataclass describing those
parameters and helper functions to populate the configuration from
environment variables.  All parameters have reasonable defaults so the
module can be used without any external configuration, while still
allowing users to override values at runtime by setting environment
variables with the ``ECHOPRESS_`` prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class EchoPressConfig:
    """Container for user configurable parameters.

    Attributes
    ----------
    alpha, beta:
        Lists of calibration coefficients such that
        ``pressure = alpha[k] * voltage + beta[k]`` for channel ``k``.
    scalar_channel:
        Default channel index used when a scalar voltage value is
        supplied to the calibration routine.
    O_max:
        Maximum allowed absolute alignment error when mapping the
        streams.  ``None`` disables the check.
    tie_break:
        Strategy for resolving ties when multiple P-stream timestamps are
        equally close to an O-stream midpoint.  Supported values are
        ``"nearest"`` (first match), ``"first"`` and ``"last"``.
    window_size:
        Number of neighbouring timestamps in the P-stream to consider
        when searching for the closest match.
    kappa:
        Multiplicative factor applied to the alignment error ``E_align``.
    """

    alpha: List[float] = field(default_factory=list)
    beta: List[float] = field(default_factory=list)
    scalar_channel: int = 0
    O_max: Optional[float] = None
    tie_break: str = "nearest"
    window_size: int = 5
    kappa: float = 1.0


ENV_PREFIX = "ECHOPRESS_"


def _get_env_list(name: str) -> List[float]:
    value = os.getenv(ENV_PREFIX + name)
    if not value:
        return []
    return [float(v) for v in value.split(",") if v]


def load_config() -> EchoPressConfig:
    """Create a configuration instance populated from environment variables."""
    cfg = EchoPressConfig()
    alpha = _get_env_list("ALPHA")
    beta = _get_env_list("BETA")
    if alpha:
        cfg.alpha = alpha
    if beta:
        cfg.beta = beta

    if (scalar := os.getenv(ENV_PREFIX + "SCALAR_CHANNEL")) is not None:
        cfg.scalar_channel = int(scalar)
    if (o_max := os.getenv(ENV_PREFIX + "O_MAX")) is not None:
        cfg.O_max = float(o_max)
    if (tie := os.getenv(ENV_PREFIX + "TIE_BREAK")) is not None:
        cfg.tie_break = tie
    if (window := os.getenv(ENV_PREFIX + "WINDOW_SIZE")) is not None:
        cfg.window_size = int(window)
    if (kappa := os.getenv(ENV_PREFIX + "KAPPA")) is not None:
        cfg.kappa = float(kappa)
    return cfg


_default_config: Optional[EchoPressConfig] = None


def get_config() -> EchoPressConfig:
    """Return the module level configuration instance.

    The configuration is loaded on first use and cached subsequently.  A
    copy of the configuration can be obtained by calling :func:`load_config`
    directly if isolation is desired.
    """

    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config
