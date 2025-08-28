import json
import numpy as np
from typer.testing import CliRunner
from omegaconf import OmegaConf

from echopress.cli import app


def make_cfg(tmp_path):
    o_path = tmp_path / "s1.npz"
    np.savez(
        o_path,
        session_id="s1",
        timestamps=np.array([0.0, 1.0, 2.0]),
        channels=np.array([1.0, 2.0, 3.0]),
    )
    p_path = tmp_path / "voltprsr001.csv"
    p_path.write_text("timestamp,pressure\n0,10\n1,11\n2,12\n")
    align_path = tmp_path / "align.json"
    cfg = OmegaConf.create(
        {
            "dataset": {
                "root": {"ostream": str(tmp_path), "pstream": str(tmp_path)},
                "ostream": str(o_path),
                "pstream": str(p_path),
            },
            "mapping": {"tie_breaker": "earliest", "O_max": 2.0, "W": 5, "kappa": 1.0},
            "quality": {"reject_if_Ealign_gt_Omax": True, "min_records_in_W": 1},
            "calibration": {"alpha": [1, 1, 1], "beta": [0, 0, 0]},
            "pressure": {"scalar_channel": 2},
            "units": {"pressure": "Pa", "voltage": "V"},
            "timestamp": {"timezone": "UTC"},
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
    return cfg, o_path, p_path, align_path


def test_index_align_adapt(tmp_path):
    cfg, o_path, p_path, align_path = make_cfg(tmp_path)
    runner = CliRunner()

    cache_path = tmp_path / "index.json"
    result = runner.invoke(app, ["index", "--cache", str(cache_path)], obj=cfg)
    assert result.exit_code == 0
    data = json.loads(cache_path.read_text())
    assert "s1" in data["ostreams"]
    assert "voltprsr001" in data["pstreams"]

    result = runner.invoke(app, ["align", "--export", str(align_path)], obj=cfg)
    assert result.exit_code == 0
    data = json.loads(align_path.read_text())
    assert any("pressure_value" in row for row in data)

    out_path = tmp_path / "features.npy"
    result = runner.invoke(
        app,
        ["adapt", "--adapter", "cec", "--n", "1", "--output", str(out_path)],
        obj=cfg,
    )
    assert result.exit_code == 0
    assert "adapter=cec" in result.stdout
    assert out_path.exists()
    data = np.load(out_path)
    assert data.shape[0] == 1
