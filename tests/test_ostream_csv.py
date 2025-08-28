import numpy as np
from echopress.ingest import load_ostream, DatasetIndexer


def test_load_ostream_csv(tmp_path):
    csv_path = tmp_path / "s1.csv"
    csv_path.write_text("\n".join([
        "session_id,timestamp,ch0,ch1",
        "s1,0.0,1.0,2.0",
        "s1,1.0,3.0,4.0",
    ]))
    ostream = load_ostream(csv_path)
    assert ostream.session_id == "s1"
    np.testing.assert_allclose(ostream.timestamps, [0.0, 1.0])
    np.testing.assert_allclose(ostream.channels, [[1.0, 2.0], [3.0, 4.0]])
    assert ostream.meta == {}


def test_dataset_indexer_picks_up_csv(tmp_path):
    csv_path = tmp_path / "sessionA.csv"
    csv_path.write_text("session_id,timestamp,ch0\nsessionA,0.0,1.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert "sessionA" in indexer.ostreams
    assert csv_path in indexer.ostreams["sessionA"]
    assert indexer.get_ostreams("sessionA") == [csv_path]
    assert indexer.get_ostreams("SESSIONA") == [csv_path]
