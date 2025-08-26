from typing import Dict
import numpy as np

from ..base import (
    Adapter,
    register,
    cycle_sync_mapping,
    ft_spectrum,
    hilbert_envelope,
    wavelet_energies,
    mfcc_features,
)


@register("cec")
class AdapterImpl:
    """Adapter implementation for cec."""

    @staticmethod
    def layer1(signal: np.ndarray, cycle_indices: np.ndarray) -> np.ndarray:
        return cycle_sync_mapping(signal, cycle_indices)

    @staticmethod
    def layer2(signal: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
        return {
            "ft": ft_spectrum(signal),
            "hilbert": hilbert_envelope(signal),
            "wavelet": wavelet_energies(signal),
            "mfcc": mfcc_features(signal, fs),
        }
