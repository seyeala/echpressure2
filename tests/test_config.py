import json
import pytest

from echopress.config import Settings, load_settings


def test_from_env(monkeypatch):
    monkeypatch.setenv("ECHOPRESS_ALPHA", "2.5")
    monkeypatch.setenv("ECHOPRESS_REJECT_IF_EALIGN_GT_OMAX", "false")
    s = Settings.from_env()
    assert s.alpha == 2.5
    assert s.reject_if_Ealign_gt_Omax is False


def test_load_settings_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"beta": 1.2, "W": 7}))
    s = load_settings(p)
    assert s.beta == 1.2
    assert s.W == 7


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@pytest.mark.skipif(yaml is None, reason="PyYAML not installed")
def test_load_settings_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("alpha: 1.5\nW: 9\n")
    s = load_settings(p)
    assert s.alpha == 1.5
    assert s.W == 9
