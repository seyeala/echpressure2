from __future__ import annotations

"""Adapter protocol, registry and core transforms."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Dict, List, Callable
import numpy as np


@runtime_checkable
class Adapter(Protocol):
    """Protocol describing an adapter.

    Adapters expose two layers: ``layer1`` performs cycle-synchronous
    mapping while ``layer2`` applies signal transforms to the mapped
    cycles.  Both layers operate on :class:`numpy.ndarray` objects.
    """

    name: str

    def layer1(self, signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
        """Map ``signal`` into cycle-synchronous representation.

        Parameters
        ----------
        signal:
            One-dimensional signal array.
        fs:
            Sampling frequency of ``signal`` in Hz.
        f0:
            Fundamental frequency in Hz used to determine the cycle
            duration.
        """

    def layer2(self, cycles: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
        """Apply transforms to ``cycles`` and return a dictionary of
        named outputs."""


_registry: Dict[str, Adapter] = {}


def register_adapter(adapter: Adapter) -> None:
    """Register ``adapter`` in the global registry."""
    validate_adapter(adapter)
    _registry[adapter.name] = adapter


def get_adapter(name: str) -> Adapter:
    """Retrieve an adapter by ``name``."""
    return _registry[name]


def available_adapters() -> List[str]:
    """Return the list of registered adapter names."""
    return list(_registry)


def validate_adapter(adapter: Adapter) -> None:
    """Validate that ``adapter`` satisfies the :class:`Adapter` protocol."""
    if not isinstance(adapter, Adapter):
        raise TypeError("Adapter does not implement the required protocol")


# ---------------------------------------------------------------------------
# Layer-1 mapping utilities
# ---------------------------------------------------------------------------

def cycle_synchronous_map(signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """Segment ``signal`` into cycle-synchronous slices.

    The signal is reshaped into ``(n_cycles, cycle_len)`` where
    ``cycle_len`` is determined from ``fs`` and ``f0``.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    cycle_len = int(fs / f0)
    if cycle_len <= 0:
        raise ValueError("cycle length must be positive")
    n_cycles = signal.size // cycle_len
    if n_cycles == 0:
        raise ValueError("signal too short for a single cycle")
    trimmed = signal[: n_cycles * cycle_len]
    return trimmed.reshape(n_cycles, cycle_len)


# ---------------------------------------------------------------------------
# Layer-2 transforms
# ---------------------------------------------------------------------------

def ft_spectrum(cycles: np.ndarray) -> np.ndarray:
    """Return the magnitude Fourier spectrum for each cycle."""
    return np.abs(np.fft.rfft(cycles, axis=-1))


def hilbert_envelope(cycles: np.ndarray) -> np.ndarray:
    """Return the Hilbert envelope for each cycle.

    Uses an FFT-based implementation of the analytic signal to avoid the
    SciPy dependency.
    """
    n = cycles.shape[-1]
    fft = np.fft.fft(cycles, axis=-1)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = np.fft.ifft(fft * h, axis=-1)
    return np.abs(analytic)


def wavelet_energies(cycles: np.ndarray) -> np.ndarray:
    """Compute simple Haar wavelet energy for each cycle."""
    if cycles.shape[-1] % 2 == 1:
        cycles = cycles[..., :-1]
    a = (cycles[..., ::2] + cycles[..., 1::2]) / 2.0
    d = (cycles[..., ::2] - cycles[..., 1::2]) / 2.0
    energy_a = np.sum(a ** 2, axis=-1)
    energy_d = np.sum(d ** 2, axis=-1)
    return np.stack([energy_a, energy_d], axis=-1)


def _dct_matrix(N: int, K: int) -> np.ndarray:
    k = np.arange(N)
    m = np.arange(K)[:, None]
    return np.cos(np.pi * (k + 0.5) * m / N)


def mfcc(cycles: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Compute a very small MFCC approximation.

    This implementation performs a log-magnitude spectrum followed by a
    type-II DCT using a direct matrix formulation.  It is lightweight and
    avoids external dependencies but is sufficient for testing purposes.
    """
    mag = np.abs(np.fft.rfft(cycles, axis=-1))
    log_mag = np.log(mag + 1e-12)
    N = log_mag.shape[-1]
    K = min(n_mfcc, N)
    dct_mat = _dct_matrix(N, K)
    coeffs = log_mag @ dct_mat.T
    return coeffs
