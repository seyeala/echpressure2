import json
import pytest
from echopress.core.config_io import load_yaml_defaults, merge_config, write_resolved_config

from echopress.config import Settings, load_settings


def test_from_env(monkeypatch):
    monkeypatch.setenv("ECHOPRESS_CALIBRATION__ALPHA", "2.5")
    s = Settings()
    assert s.calibration.alpha == [2.5]


def test_from_env_ingest_patterns(monkeypatch):
    monkeypatch.setenv("ECHOPRESS_INGEST__PSTREAM_CSV_PATTERNS", "foo,bar")
    s = Settings()
    assert s.ingest.pstream_csv_patterns == ["foo", "bar"]


def test_load_settings_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"calibration": {"beta": [1.2]}, "mapping": {"W": 7}}))
    s = load_settings(p)
    assert s.calibration.beta[0] == 1.2
    assert s.mapping.W == 7


try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover
    _yaml = None


@pytest.mark.skipif(_yaml is None, reason="PyYAML not installed")
def test_load_settings_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("calibration:\n  alpha: [1.5]\nmapping:\n  W: 9\n")
    s = load_settings(p)
    assert s.calibration.alpha[0] == 1.5
    assert s.mapping.W == 9


@pytest.mark.skipif(_yaml is None, reason="PyYAML not installed")
def test_merge_config_precedence_and_write(tmp_path):
    defaults = tmp_path / "defaults.yml"
    user = tmp_path / "user.yml"
    defaults.write_text("a: 1\nb: 2\n")
    user.write_text("b: 30\nc: 40\n")

    resolved = merge_config(
        default_yaml_path=defaults,
        user_yaml_path=user,
        cli_values={"c": 400, "d": 500, "e": None},
    )

    assert resolved == {"a": 1, "b": 30, "c": 400, "d": 500}

    out = tmp_path / "resolved.yml"
    write_resolved_config(resolved, out)
    loaded = load_yaml_defaults(out)
    assert loaded == resolved
