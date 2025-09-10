import json
import numpy as np
from typer.testing import CliRunner

from echopress.cli import app
from test_cli_commands import make_cfg


def test_align_skips_zero_channel_ostream(tmp_path):
    cfg, align_path = make_cfg(tmp_path)
    # Remove default files created by make_cfg
    (tmp_path / "s1.npz").unlink()
    (tmp_path / "voltprsr001.csv").unlink()

    # Create an O-stream with only timestamps (zero channels)
    o_path = tmp_path / "s0.npz"
    np.savez(
        o_path,
        session_id="s0",
        timestamps=np.array([0.0, 1.0, 2.0]),
        channels=np.empty((3, 0)),
    )
    # Corresponding P-stream
    p_path = tmp_path / "voltprsr002.csv"
    p_path.write_text("timestamp,pressure\n0,10\n1,11\n2,12\n")

    runner = CliRunner()
    result = runner.invoke(app, ["align", str(tmp_path)], obj=cfg)
    assert result.exit_code == 0
    data = json.loads(align_path.read_text())
    s0_rows = [row for row in data if row["sid"] == "s0"]
    assert s0_rows
    row = s0_rows[0]
    assert "pressure_value" in row
    assert "value" not in row
