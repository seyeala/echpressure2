from echopress.ingest import DatasetIndexer


def test_dataset_indexer_picks_up_voltprsr_csv(tmp_path):
    csv_path = tmp_path / "voltprsr001.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert "voltprsr001" in indexer.pstreams
    assert csv_path in indexer.pstreams["voltprsr001"]
    assert "voltprsr001" not in indexer.ostreams


def test_dataset_indexer_picks_up_mixedcase_voltprsr_csv(tmp_path):
    csv_path = tmp_path / "VoltPrsr001.csv"
    csv_path.write_text("timestamp\n0.0\n")
    indexer = DatasetIndexer(tmp_path)
    assert "VoltPrsr001" in indexer.pstreams
    assert csv_path in indexer.pstreams["VoltPrsr001"]
    assert "VoltPrsr001" not in indexer.ostreams
