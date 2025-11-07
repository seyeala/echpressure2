import json
from typer.testing import CliRunner

from echopress.cli import app
from test_cli_commands import make_cfg


def test_align_window_mode(tmp_path):
    cfg, align_path = make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "mapping": cfg.mapping.model_copy(update={"O_max": 0.0001}),
            "quality": cfg.quality.model_copy(
                update={"reject_if_Ealign_gt_Omax": True}
            ),
        }
    )
    # remove default files from make_cfg
    (tmp_path / "s1.npz").unlink()
    (tmp_path / "voltprsr001.csv").unlink()

    o_path = tmp_path / "M01-D01-H00-M00-S00-U.000.os"
    o_path.write_text("")
    p_path = tmp_path / "voltprsr001.csv"
    p_path.write_text("timestamp,pressure\n0,9\n0.02,10\n0.04,11\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "align",
            str(tmp_path),
            "--window-mode",
            "--duration",
            "0.04",
            "--base-year",
            "1970",
        ],
        obj=cfg,
    )
    assert result.exit_code == 0
    assert "window mode" in result.stdout.lower()
    data = json.loads(align_path.read_text())
    rows = [row for row in data if row["sid"] == o_path.stem]
    assert rows
    row = rows[0]
    assert row["path"] == str(o_path)
    assert row["pressure_value"] == 10.0
    assert "value" not in row
