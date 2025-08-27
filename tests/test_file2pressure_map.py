import pytest

from echopress.core.tables import Signals, OscFiles, File2PressureMap, export_tables


def test_single_pressure_label_per_file():
    signals = Signals()
    signals.add("s", "f", 0, 1.0)
    signals.add("s", "f", 1, 2.0)

    osc = OscFiles()
    osc.add("s", "f", 0, "a")
    osc.add("s", "f", 1, "b")

    fmap = File2PressureMap()
    fmap.add("s", "f", "P")

    tall = export_tables(signals, osc, fmap, tall=True)
    labels = {row["idx"]: row["pressure_label"] for row in tall}
    assert labels == {0: "P", 1: "P"}

    with pytest.raises(KeyError):
        fmap.add("s", "f", "Q")
