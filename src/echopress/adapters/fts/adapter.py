from ..base import (
    Adapter,
    register_adapter,
    cycle_synchronous_map,
    ft_spectrum,
    hilbert_envelope,
    wavelet_energies,
    mfcc as mfcc_transform,
)
import numpy as np


class FtsAdapter:
    name = "fts"

    def layer1(self, signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
        return cycle_synchronous_map(signal, fs, f0)

    def layer2(self, cycles: np.ndarray, fs: float):
        if self.name == "fts":
            return {"spectrum": ft_spectrum(cycles)}
        if self.name == "hte":
            return {"envelope": hilbert_envelope(cycles)}
        if self.name == "wcv":
            return {"energies": wavelet_energies(cycles)}
        if self.name == "mfcc":
            return {"mfcc": mfcc_transform(cycles)}
        return {"cycles": cycles}


register_adapter(FtsAdapter())
