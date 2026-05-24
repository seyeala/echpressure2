import numpy as np

from echopress.core.signatures import extract_peak_centered, load_signature_row, write_signature_chunks


def test_peak_centered_extract_zero_pads_edges():
    x = np.array([1.0, 2.0, 3.0])
    y = extract_peak_centered(x, 0, left=2, right=2)
    assert y.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]


def test_chunk_write_and_indexed_load(tmp_path):
    sig = np.arange(30.0).reshape(10, 3)
    index = write_signature_chunks(sig, tmp_path, chunk_size=4)
    row = load_signature_row(index, 7)
    assert np.allclose(row, sig[7])
