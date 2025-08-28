from echopress.ingest import DatasetIndexer
from echopress.config import Settings


def test_dataset_indexer_picks_up_voltprsr_csv(tmp_path):
    csv_path = tmp_path / "voltprsr001.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert "001" in indexer.pstreams
    assert csv_path in indexer.pstreams["001"]
    assert "001" not in indexer.ostreams


def test_dataset_indexer_picks_up_multiple_patterns(tmp_path):
    (tmp_path / "voltprsr001.csv").write_text("timestamp\n0.0\n")
    (tmp_path / "anotherpstream002.csv").write_text("timestamp\n0.0\n")
    (tmp_path / "sessionA.csv").write_text(
        "session_id,timestamp,ch0\nsessionA,0.0,1.0\n"
    )
    settings = Settings()
    settings.ingest.pstream_csv_patterns = ["voltprsr", "anotherpstream"]
    indexer = DatasetIndexer(tmp_path, settings=settings)
    assert "001" in indexer.pstreams
    assert "002" in indexer.pstreams
    assert "sessiona" in indexer.ostreams
    assert "sessiona" not in indexer.pstreams


def test_dataset_indexer_lookup_by_stripped_id(tmp_path):
    csv_path = tmp_path / "VoltPrsr001.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert indexer.get_pstreams("001") == [csv_path]


def test_dataset_indexer_accepts_regex_patterns(tmp_path):
    csv_path = tmp_path / "VoltPrsr123.csv"
    csv_path.write_text("timestamp\n0.0\n")
    settings = Settings()
    settings.ingest.pstream_csv_patterns = [r"voltprsr\d+"]
    indexer = DatasetIndexer(tmp_path, settings=settings)
    assert "voltprsr123" in indexer.pstreams
    assert csv_path in indexer.pstreams["voltprsr123"]


def test_dataset_indexer_handles_prefix_only(tmp_path):
    csv_path = tmp_path / "voltprsr.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert "" not in indexer.pstreams
    assert "voltprsr" in indexer.pstreams
    assert csv_path in indexer.pstreams["voltprsr"]

