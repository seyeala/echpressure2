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
from ...config import Settings
import numpy as np
import warnings


class PlstnAdapter:
    name = "plstn"

    def __init__(
        self,
        window_left: float | int | None = None,
        window_right: float | int | None = None,
        resample_len: int | None = None,
    ) -> None:
        self.window_left = window_left
        self.window_right = window_right
        self.resample_len = resample_len

    @staticmethod
    def _resolve_window_size(value: float | int | None, cycle_len: int, default_frac: float) -> int:
        if value is None:
            size = int(round(default_frac * cycle_len))
        elif isinstance(value, float) and value <= 1.0:
            size = int(round(value * cycle_len))
        else:
            size = int(value)
        if size < 0:
            raise ValueError("window size must be non-negative")
        return size

    def layer1(self, signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        cycle_len = int(fs / f0)
        if cycle_len <= 0:
            raise ValueError("cycle length must be positive")
        settings = Settings()
        if self.window_left is None:
            self.window_left = settings.adapter.plstn.window_left
        if self.window_right is None:
            self.window_right = settings.adapter.plstn.window_right
        if self.resample_len is None:
            self.resample_len = settings.adapter.plstn.resample_len
        w_left = self._resolve_window_size(self.window_left, cycle_len, default_frac=0.5)
        w_right = self._resolve_window_size(self.window_right, cycle_len, default_frac=0.5)
        window_len = w_left + w_right
        if window_len <= 0:
            raise ValueError("window length must be positive")
        resample_len = window_len if self.resample_len is None else int(self.resample_len)
        if resample_len <= 0:
            raise ValueError("resample length must be positive")
        if signal.size < window_len:
            warnings.warn(
                "Signal shorter than peak window length; falling back to fixed cycle slicing.",
                RuntimeWarning,
            )
            return cycle_synchronous_map(signal, fs, f0)
        min_distance = max(1, int(0.8 * cycle_len))
        peaks = detect_peaks(signal, min_distance)
        if peaks.size < 1:
            warnings.warn(
                "Too few peaks detected for peak-locked windows; falling back to "
                "fixed cycle slicing.",
                RuntimeWarning,
            )
            return cycle_synchronous_map(signal, fs, f0)
        windows = []
        for peak_idx in peaks:
            start_idx = peak_idx - w_left
            end_idx = start_idx + window_len
            if start_idx < 0:
                start_idx = 0
                end_idx = window_len
            if end_idx > signal.size:
                end_idx = signal.size
                start_idx = end_idx - window_len
            if start_idx < 0 or end_idx > signal.size:
                continue
            segment = signal[start_idx:end_idx]
            if segment.size < 1:
                continue
            if segment.size == 1:
                resampled = np.full(resample_len, segment[0], dtype=segment.dtype)
            elif segment.size == resample_len:
                resampled = segment.astype(float, copy=False)
            else:
                source_idx = np.linspace(0.0, 1.0, num=segment.size)
                target_idx = np.linspace(0.0, 1.0, num=resample_len)
                resampled = np.interp(target_idx, source_idx, segment).astype(float, copy=False)
            windows.append(resampled)
        if not windows:
            warnings.warn(
                "No valid peak-locked windows found; falling back to fixed "
                "cycle slicing.",
                RuntimeWarning,
            )
            return cycle_synchronous_map(signal, fs, f0)
        return np.stack(windows, axis=0)

    def layer2(self, cycles: np.ndarray, fs: float):
        if self.name == "fts":
            return {"spectrum": ft_spectrum(cycles)}
        if self.name == "hte":
            return {"envelope": hilbert_envelope(cycles)}
        if self.name == "wcv":
            return {"energies": wavelet_energies(cycles)}
        if self.name == "mfcc":
            return {"mfcc": mfcc_transform(cycles)}
        if cycles.ndim == 1:
            x_f = cycles.astype(float, copy=False)
        else:
            x_f = np.median(cycles, axis=0)
        return {"x_F": x_f}


register_adapter(PlstnAdapter())
