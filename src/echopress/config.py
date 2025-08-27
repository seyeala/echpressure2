from __future__ import annotations

"""Configuration utilities for echopress."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class Settings:
    """Container for runtime configuration.

    Parameters correspond to commonly used options throughout the library.
    Values may be provided directly, via environment variables or loaded from
    a configuration file using :func:`load_settings`.
    """

    alpha: float = 1.0
    beta: float = 0.0
    channel: int = 0
    O_max: float = 0.1
    tie_break: str = "earlier"
    W: int = 5
    kappa: float = 1.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from ``ECHOPRESS_*`` environment variables."""

        mapping = {
            "alpha": ("ECHOPRESS_ALPHA", float),
            "beta": ("ECHOPRESS_BETA", float),
            "channel": ("ECHOPRESS_CHANNEL", int),
            "O_max": ("ECHOPRESS_O_MAX", float),
            "tie_break": ("ECHOPRESS_TIE_BREAK", str),
            "W": ("ECHOPRESS_W", int),
            "kappa": ("ECHOPRESS_KAPPA", float),
        }
        data: Dict[str, Any] = {}
        for field, (env, conv) in mapping.items():
            if env in os.environ:
                data[field] = conv(os.environ[env])
        return cls(**data)


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
    return Settings(**data)
