from datetime import datetime, timezone

from echopress.ingest import DatasetIndexer, read_pstream


def test_indexer_finds_and_reads_voltprsr_example(tmp_path):
    data = "timestamp,pressure\n0.0,1.0\n1.0,2.0\n"
    file = tmp_path / "voltprsr_example.csv"
    file.write_text(data)

    indexer = DatasetIndexer(tmp_path)
    assert "_example" in indexer.pstreams

    indexed_file = indexer.first_pstream("_example")
    assert indexed_file == file

    records = list(read_pstream(indexed_file))
    assert [r.pressure for r in records] == [1.0, 2.0]
    assert records[0].timestamp == datetime.fromtimestamp(0.0, tz=timezone.utc)
    assert records[1].timestamp == datetime.fromtimestamp(1.0, tz=timezone.utc)
