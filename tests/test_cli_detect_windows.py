import json

import numpy as np
from typer.testing import CliRunner

from echopress.cli import app
from echopress.config import Settings


def _cfg(tmp_path):
    o_path = tmp_path / "s1.npz"
    np.savez(
        o_path,
        session_id="s1",
        timestamps=np.array([0.0, 1.0, 2.0]),
        channels=np.array([1.0, 2.0, 3.0]),
    )
    align_path = tmp_path / "align.json"
    align_path.write_text(json.dumps([{"path": str(o_path), "pressure_value": 10.0}]))
    return Settings.model_validate(
        {
            "dataset": {"root": str(tmp_path)},
            "mapping": {"tie_breaker": "earliest", "O_max": 2.0, "W": 5, "kappa": 1.0},
            "quality": {"reject_if_Ealign_gt_Omax": True, "min_records_in_W": 1},
            "calibration": {"alpha": [1, 1, 1], "beta": [0, 0, 0]},
            "pressure": {"scalar_channel": 2},
            "units": {"pressure": "Pa", "voltage": "V"},
            "timestamp": {"timezone": "UTC"},
            "align": {"duration": 0.02, "window_mode": False, "base_year": None},
            "adapter": {
                "name": "cec",
                "output_length": 0,
                "period_est": {"fs": 1.0, "f0": 1.0},
                "align_table": str(align_path),
                "pr_min": None,
                "pr_max": None,
                "n": 1,
                "plot": False,
                "seed": 0,
            },
            "viz": {},
        }
    )


def test_detect_windows_command_registration_and_parsing(tmp_path):
    runner = CliRunner()
    out = tmp_path / "diag.json"
    result = runner.invoke(
        app,
        [
            "detect-windows",
            "--macro-k-min",
            "3",
            "--macro-k-max",
            "9",
            "--signature-chunk-size",
            "2048",
            "--diagnostics-out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["macro_k_bounds"] == [3, 9]
    assert payload["signature"]["chunk_size"] == 2048
    assert out.exists()


def test_adapt_segmentation_mode_dispatch_and_legacy_flag(tmp_path):
    cfg = _cfg(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["adapt", "--segmentation", "none", "--n", "1"],
        obj=cfg,
    )
    assert result.exit_code == 0

    legacy = runner.invoke(
        app,
        ["adapt", "--use-rmcpe-tciml", "--n", "1"],
        obj=cfg,
    )
    assert legacy.exit_code == 0
    assert "Deprecation warning" in legacy.stdout
