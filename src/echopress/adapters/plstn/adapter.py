from ..base import register_adapter, cycle_synchronous_map
import numpy as np


class PlstnAdapter:
    name = "plstn"

    @staticmethod
    def _detect_peaks(signal: np.ndarray, min_distance: int) -> np.ndarray:
        if signal.size < 3:
            return np.array([], dtype=int)
        candidates = np.where(
            (signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])
        )[0] + 1
        if candidates.size == 0:
            return np.array([], dtype=int)
        selected = [int(candidates[0])]
        for idx in candidates[1:]:
            idx = int(idx)
            if idx - selected[-1] >= min_distance:
                selected.append(idx)
            elif signal[idx] > signal[selected[-1]]:
                selected[-1] = idx
        return np.array(selected, dtype=int)

    @staticmethod
    def _resample(window: np.ndarray, target_len: int) -> np.ndarray:
        if window.size == 0:
            return np.zeros(target_len)
        if window.size == target_len:
            return window
        x_old = np.linspace(0.0, 1.0, num=window.size, endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
        return np.interp(x_new, x_old, window)

    def layer1(self, signal: np.ndarray, fs: float, f0: float) -> np.ndarray:
        cycle_len = int(fs / f0)
        if cycle_len <= 0:
            raise ValueError("cycle length must be positive")
        min_distance = max(1, int(0.8 * cycle_len))
        peaks = self._detect_peaks(signal, min_distance=min_distance)
        if peaks.size == 0:
            return cycle_synchronous_map(signal, fs, f0)
        w_left = cycle_len // 2
        w_right = cycle_len - w_left
        windows = []
        for peak in peaks:
            start = peak - w_left
            end = peak + w_right
            if start < 0 or end > signal.size:
                continue
            window = signal[start:end]
            windows.append(self._resample(window, cycle_len))
        if not windows:
            return cycle_synchronous_map(signal, fs, f0)
        return np.stack(windows)

    def layer2(self, cycles: np.ndarray, fs: float):
        if cycles.ndim == 1:
            cycles = cycles[None, :]
        avg = np.median(cycles, axis=0)
        return {"plstn": avg}


register_adapter(PlstnAdapter())
