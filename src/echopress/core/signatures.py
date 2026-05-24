from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_peak_centered(signal: np.ndarray, peak_idx: int, *, left: int, right: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    width = left + right + 1
    out = np.zeros(width, dtype=float)
    start = int(peak_idx) - left
    stop = int(peak_idx) + right + 1
    src_lo = max(0, start)
    src_hi = min(x.size, stop)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    if src_hi > src_lo:
        out[dst_lo:dst_hi] = x[src_lo:src_hi]
    return out


def write_signature_chunks(signatures: np.ndarray, out_dir: str | Path, *, chunk_size: int = 128, stem: str = "signatures") -> Path:
    arr = np.asarray(signatures, dtype=float)
    if arr.ndim != 2:
        raise ValueError("signatures must be 2D")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(0, arr.shape[0], chunk_size):
        chunk = arr[i : i + chunk_size]
        name = f"{stem}_{i//chunk_size:04d}.npy"
        np.save(out / name, chunk)
        records.append((i, i + chunk.shape[0], name))

    idx = np.array(records, dtype=object)
    index_path = out / f"{stem}_index.npy"
    np.save(index_path, idx, allow_pickle=True)
    return index_path


def load_signature_row(index_path: str | Path, row_index: int) -> np.ndarray:
    idx = np.load(index_path, allow_pickle=True)
    base = Path(index_path).parent
    for start, end, fname in idx:
        s, e = int(start), int(end)
        if s <= row_index < e:
            data = np.load(base / str(fname))
            return data[row_index - s]
    raise IndexError(f"row index out of range: {row_index}")


__all__ = ["extract_peak_centered", "write_signature_chunks", "load_signature_row"]
