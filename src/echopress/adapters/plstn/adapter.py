from ..base import Adapter, register_adapter
import numpy as np


def _find_peaks(signal: np.ndarray, min_distance: int) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    candidate_idx = np.where(
        (signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])
    )[0] + 1
    if candidate_idx.size == 0:
        return np.array([], dtype=int)
    if min_distance <= 1:
        return candidate_idx
    peaks = [candidate_idx[0]]
    last = candidate_idx[0]
    for idx in candidate_idx[1:]:
        if idx - last >= min_distance:
            peaks.append(idx)
            last = idx
    return np.asarray(peaks, dtype=int)


def _extract_windows(
    signal: np.ndarray, peaks: np.ndarray, left: int, right: int
) -> np.ndarray:
    window_len = left + right
    windows = []
    for peak in peaks:
        start = peak - left
        end = peak + right
        if start < 0 or end > signal.size:
            continue
        windows.append(signal[start:end])
    if not windows:
        return np.empty((0, window_len), dtype=signal.dtype)
    return np.stack(windows, axis=0)


def _resample_windows(windows: np.ndarray, target_len: int) -> np.ndarray:
    if windows.size == 0:
        return windows.reshape(0, target_len)
    n_windows, source_len = windows.shape
    if source_len == target_len:
        return windows
    x_old = np.linspace(0.0, 1.0, source_len)
    x_new = np.linspace(0.0, 1.0, target_len)
    resampled = np.empty((n_windows, target_len), dtype=windows.dtype)
    for idx, window in enumerate(windows):
        resampled[idx] = np.interp(x_new, x_old, window)
    return resampled


class PlstnAdapter:
    name = "plstn"

    def layer1(self, signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
        cycle_len = int(fs / f0)
        if cycle_len <= 0:
            raise ValueError("cycle length must be positive")
        min_distance = max(1, int(0.8 * cycle_len))
        peaks = _find_peaks(signal, min_distance=min_distance)
        left = max(1, cycle_len // 2)
        right = max(1, cycle_len - left)
        windows = _extract_windows(signal, peaks, left=left, right=right)
        return _resample_windows(windows, target_len=cycle_len)

    def layer2(self, cycles: np.ndarray, fs: float):
        if cycles.size == 0:
            return {"plstn": np.array([])}
        return {"plstn": np.median(cycles, axis=0)}


register_adapter(PlstnAdapter())
