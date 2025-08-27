import pytest

from echopress.core.tables import Signals, OscFiles, File2PressureMap, export_tables


def test_single_pressure_value_per_file():
    signals = Signals()
    signals.add("s", "f", 0, 1.0)
    signals.add("s", "f", 1, 2.0)

    osc = OscFiles()
    osc.add("s", "f", 0, "a")
    osc.add("s", "f", 1, "b")

    fmap = File2PressureMap()
    fmap.add("s", "f", 100.0, alignment_error=0.5)

    tall = export_tables(signals, osc, fmap, tall=True)
    labels = {row["idx"]: row["pressure_value"] for row in tall}
    assert labels == {0: 100.0, 1: 100.0}
    aligns = {row["idx"]: row["alignment_error"] for row in tall}
    assert aligns == {0: 0.5, 1: 0.5}

    with pytest.raises(KeyError):
        fmap.add("s", "f", 101.0)
