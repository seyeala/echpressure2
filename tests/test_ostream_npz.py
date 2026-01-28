import numpy as np
from echopress.ingest import load_ostream


def test_load_ostream_npz_mV_time_ns(tmp_path):
    npz_path = tmp_path / "s1.npz"
    mV = np.array([1.0, 2.0, 3.0], dtype=float)
    time_ns = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    np.savez(npz_path, session_id="s1", mV=mV, time_ns=time_ns)

    ostream = load_ostream(npz_path)

    assert ostream.session_id == "s1"
    np.testing.assert_allclose(ostream.channels, mV.reshape(-1, 1))
    np.testing.assert_allclose(ostream.timestamps, time_ns / 1e9)
    assert ostream.meta["channels_source"] == "mV"
