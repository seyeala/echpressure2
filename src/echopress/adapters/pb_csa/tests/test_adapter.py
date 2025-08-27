from echopress.adapters import get_adapter
import numpy as np


def test_registry():
    adapter = get_adapter("pb_csa")
    assert adapter.name == "pb_csa"


def test_layer1_and_layer2():
    adapter = get_adapter("pb_csa")
    signal = np.arange(20, dtype=float)
    cycles = adapter.layer1(signal, fs=10.0, f0=2.0)
    assert cycles.ndim == 2
    out = adapter.layer2(cycles, fs=10.0)
    assert isinstance(out, dict)
