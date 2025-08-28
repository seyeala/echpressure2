import json
import pytest

from echopress.config import Settings, load_settings


def test_from_env(monkeypatch):
    monkeypatch.setenv("ECHOPRESS_CALIBRATION_ALPHA", "2.5")
    s = Settings.from_env()
    assert s.calibration.alpha[0] == 2.5


def test_from_env_ingest_patterns(monkeypatch):
    monkeypatch.setenv("ECHOPRESS_INGEST_PSTREAM_CSV_PATTERNS", "foo,bar")
    s = Settings.from_env()
    assert s.ingest.pstream_csv_patterns == ["foo", "bar"]


def test_load_settings_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"calibration": {"beta": [1.2]}, "mapping": {"W": 7}}))
    s = load_settings(p)
    assert s.calibration.beta[0] == 1.2
    assert s.mapping.W == 7


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@pytest.mark.skipif(yaml is None, reason="PyYAML not installed")
def test_load_settings_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("calibration:\n  alpha: [1.5]\nmapping:\n  W: 9\n")
    s = load_settings(p)
    assert s.calibration.alpha[0] == 1.5
    assert s.mapping.W == 9
