import pytest

from echopress.core import load_config


def test_load_config(tmp_path):
    text = (
        "calibration:\n"
        "  alpha: [1, 2, 3]\n"
        "  beta: [0, 0, 0]\n"
        "pressure:\n"
        "  scalar_channel: 2\n"
        "mapping:\n"
        "  O_max: 0.5\n"
        "  tie_breaker: latest\n"
        "derivative:\n"
        "  W: 5\n"
        "uncertainty:\n"
        "  kappa: 0.5\n"
    )
    path = tmp_path / "cfg.yml"
    path.write_text(text)
    cfg = load_config(path)
    assert cfg.calibration.alpha == [1, 2, 3]
    assert cfg.calibration.beta == [0, 0, 0]
    assert cfg.pressure.scalar_channel == 2
    assert cfg.mapping.O_max == pytest.approx(0.5)
    assert cfg.mapping.tie_breaker == "latest"
    assert cfg.derivative.W == 5
    assert cfg.uncertainty.kappa == pytest.approx(0.5)
