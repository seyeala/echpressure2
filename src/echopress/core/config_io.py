from __future__ import annotations

from pathlib import Path
from typing import Any


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


def write_resolved_config(config: dict[str, Any], path: Path) -> None:
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
