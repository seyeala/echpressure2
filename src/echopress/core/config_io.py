from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for YAML config support") from exc
    return yaml


def load_yaml_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    yaml = _require_yaml()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return loaded


def parse_override_value(raw: str) -> Any:
    """Parse a command-line override value into a Python object.

    Values intentionally accept JSON-ish scalars and containers so CLI users can
    write overrides such as ``--set model.hidden_units=[32,16]`` while keeping
    plain strings unchanged.
    """
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON override value: {raw}") from exc
    return raw


def apply_override(data: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    """Apply a parsed value to a nested mapping using dotted-key components."""
    if not keys or any(key == "" for key in keys):
        raise ValueError("override key cannot be empty")
    target = data
    for key in keys[:-1]:
        existing = target.get(key)
        if not isinstance(existing, dict):
            existing = {}
            target[key] = existing
        target = existing
    target[keys[-1]] = value


def apply_dotted_overrides(data: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    """Return a copy of *data* with ``section.key=value`` overrides applied."""
    resolved = copy.deepcopy(data)
    if not overrides:
        return resolved
    for override in overrides:
        if "=" not in override:
            raise ValueError("overrides must be of the form --set section.key=value")
        key, raw_value = override.split("=", 1)
        keys = key.split(".")
        value = parse_override_value(raw_value)
        apply_override(resolved, keys, value)
    return resolved


def merge_config(
    *,
    default_yaml_path: Path,
    user_yaml_path: Path | None,
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    resolved = load_yaml_defaults(default_yaml_path)
    if user_yaml_path is not None:
        resolved.update(load_yaml_defaults(user_yaml_path))
    resolved.update({k: v for k, v in cli_values.items() if v is not None})
    return resolved


def make_yaml_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_yaml_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_yaml_safe(v) for v in obj]
    return obj


def write_resolved_config(config: dict[str, Any], path: Path) -> None:
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = make_yaml_safe(config)
    path.write_text(yaml.safe_dump(safe, sort_keys=False), encoding="utf-8")
