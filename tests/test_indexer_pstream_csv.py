from echopress.ingest import DatasetIndexer
from echopress.config import Settings


def test_dataset_indexer_picks_up_voltprsr_csv(tmp_path):
    csv_path = tmp_path / "voltprsr001.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert "voltprsr001" in indexer.pstreams
    assert csv_path in indexer.pstreams["voltprsr001"]
    assert "voltprsr001" not in indexer.ostreams


def test_dataset_indexer_picks_up_multiple_patterns(tmp_path):
    (tmp_path / "voltprsr001.csv").write_text("timestamp\n0.0\n")
    (tmp_path / "anotherpstream.csv").write_text("timestamp\n0.0\n")
    (tmp_path / "sessionA.csv").write_text(
        "session_id,timestamp,ch0\nsessionA,0.0,1.0\n"
    )
    settings = Settings()
    settings.pstream_csv_patterns = ["voltprsr", "anotherpstream"]
    indexer = DatasetIndexer(tmp_path, settings=settings)
    assert "voltprsr001" in indexer.pstreams
    assert "anotherpstream" in indexer.pstreams
    assert "sessiona" in indexer.ostreams
    assert "sessiona" not in indexer.pstreams


def test_dataset_indexer_case_insensitive_lookup(tmp_path):
    csv_path = tmp_path / "VoltPrsr001.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert indexer.get_pstreams("voltprsr001") == [csv_path]
