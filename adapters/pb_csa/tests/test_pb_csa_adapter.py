from __future__ import annotations
import pathlib
import sys
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from adapters.base import registry
import adapters.pb_csa.adapter  # noqa: F401


def test_adapter_registered():
    assert "pb_csa" in registry()


def test_layer_output_shapes():
    adapter = registry()["pb_csa"]
    signal = np.sin(np.linspace(0, 2 * np.pi, 64))
    cycles = np.array([0, 32, 64])
    l1 = adapter.layer1(signal, cycles)
    out = adapter.layer2(l1, 1000)
    assert out["ft"].size > 0
