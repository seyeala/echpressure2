from ..base import (
    Adapter,
    register_adapter,
    cycle_synchronous_map,
    detect_peaks,
    ft_spectrum,
    hilbert_envelope,
    wavelet_energies,
    mfcc as mfcc_transform,
)
import numpy as np
import warnings


class PlstnAdapter:
    name = "plstn"

    def layer1(self, signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        cycle_len = int(fs / f0)
        if cycle_len <= 0:
            raise ValueError("cycle length must be positive")
        min_distance = max(1, int(0.8 * cycle_len))
        peaks = detect_peaks(signal, min_distance)
        if peaks.size < 2:
            warnings.warn(
                "Too few peaks detected for cycle anchoring; falling back to "
                "fixed cycle slicing.",
                RuntimeWarning,
            )
            return cycle_synchronous_map(signal, fs, f0)
        cycles = []
        for start_idx, end_idx in zip(peaks[:-1], peaks[1:]):
            segment = signal[start_idx:end_idx]
            if segment.size >= cycle_len:
                cycles.append(segment[:cycle_len])
        if not cycles:
            warnings.warn(
                "No valid peak-to-peak cycles found; falling back to fixed "
                "cycle slicing.",
                RuntimeWarning,
            )
            return cycle_synchronous_map(signal, fs, f0)
        return np.stack(cycles, axis=0)

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


register_adapter(PlstnAdapter())
