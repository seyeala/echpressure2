from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Protocol, Type, TypeVar


class Adapter(Protocol):
    """Protocol for signal processing adapters.

    Implementations must provide two static methods:

    ``layer1(signal, cycle_indices)``
        Perform cycle synchronous mapping of ``signal`` using the provided
        ``cycle_indices`` boundaries. The indices define the start of each
        cycle and the last index marks the end of the final cycle.

    ``layer2(signal, fs)``
        Apply a secondary transform to the cycle mapped ``signal`` using the
        sampling rate ``fs``. Implementations are free to return any object but
        the built in adapters return dictionaries of numpy arrays.
    """

    @staticmethod
    def layer1(signal: np.ndarray, cycle_indices: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def layer2(signal: np.ndarray, fs: float):
        ...


# Registry -----------------------------------------------------------------

T = TypeVar("T", bound=Type[Adapter])
_registry: Dict[str, Type[Adapter]] = {}


def register(name: str) -> Callable[[T], T]:
    """Class decorator registering an :class:`Adapter` implementation.

    The decorator validates that the decorated class follows the
    :class:`Adapter` protocol and stores it in a module level registry.
    """

    def decorator(cls: T) -> T:
        validate_adapter(cls)
        _registry[name] = cls
        return cls

    return decorator


def validate_adapter(cls: Type[Adapter]) -> None:
    """Validate that ``cls`` conforms to the :class:`Adapter` protocol."""

    for attr in ("layer1", "layer2"):
        if not hasattr(cls, attr):
            raise TypeError(f"Adapter missing required attribute: {attr}")


def registry() -> Dict[str, Type[Adapter]]:
    """Return a copy of the adapter registry."""
    return dict(_registry)


# Layer‑1: cycle synchronous mapping ---------------------------------------

def cycle_sync_mapping(signal: np.ndarray, cycle_indices: np.ndarray) -> np.ndarray:
    """Return the average cycle of ``signal``.

    Parameters
    ----------
    signal:
        1-D numpy array containing the raw signal.
    cycle_indices:
        Array of indices marking the start of each cycle.  The last element
        marks the end of the final cycle.
    """

    if len(cycle_indices) < 2:
        raise ValueError("cycle_indices must contain at least two entries")

    segments = [
        signal[s:e] for s, e in zip(cycle_indices[:-1], cycle_indices[1:])
    ]
    min_len = min(len(seg) for seg in segments)
    aligned = np.array([seg[:min_len] for seg in segments])
    return aligned.mean(axis=0)


# Layer‑2: transforms -------------------------------------------------------

def ft_spectrum(signal: np.ndarray) -> np.ndarray:
    """Return magnitude of the Fourier spectrum of ``signal``."""
    return np.abs(np.fft.rfft(signal))


def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Return the Hilbert envelope of ``signal``."""
    n = len(signal)
    spectrum = np.fft.fft(signal)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * h)
    return np.abs(analytic)


def wavelet_energies(signal: np.ndarray) -> np.ndarray:
    """Compute Haar wavelet energies for ``signal``."""
    coeffs = []
    current = signal.astype(float)
    while len(current) > 1:
        approx = (current[::2] + current[1::2]) / 2.0
        detail = (current[::2] - current[1::2]) / 2.0
        coeffs.append(detail)
        current = approx
    return np.array([np.sum(c ** 2) for c in coeffs])


def mfcc_features(signal: np.ndarray, fs: float, n_mfcc: int = 13) -> np.ndarray:
    """Very small MFCC implementation using a log spectrum and DCT."""
    spectrum = np.abs(np.fft.rfft(signal))
    log_spec = np.log(spectrum + 1e-10)
    # Discrete cosine transform via an even FFT trick
    N = len(log_spec)
    extended = np.concatenate([log_spec, log_spec[::-1]])
    dct = np.real(np.fft.rfft(extended))[:N]
    return dct[:n_mfcc]


__all__ = [
    "Adapter",
    "register",
    "validate_adapter",
    "registry",
    "cycle_sync_mapping",
    "ft_spectrum",
    "hilbert_envelope",
    "wavelet_energies",
    "mfcc_features",
]
