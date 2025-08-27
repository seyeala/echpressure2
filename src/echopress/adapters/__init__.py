"""Adapter package providing signal mapping and transform utilities."""

from importlib import import_module

from .base import (
    Adapter,
    register_adapter,
    get_adapter,
    available_adapters,
    cycle_synchronous_map,
    ft_spectrum,
    hilbert_envelope,
    wavelet_energies,
    mfcc,
)

# Import adapter modules to ensure registration
for _name in [
    "pb_csa",
    "plstn",
    "hmv",
    "cec",
    "dtw_ta",
    "mtp",
    "fts",
    "hte",
    "wcv",
    "mfcc",
]:
    import_module(f".{_name}.adapter", __name__)

__all__ = [
    "Adapter",
    "register_adapter",
    "get_adapter",
    "available_adapters",
    "cycle_synchronous_map",
    "ft_spectrum",
    "hilbert_envelope",
    "wavelet_energies",
    "mfcc",
]
