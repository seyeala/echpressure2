from datetime import datetime, timezone

from echopress.ingest import read_pstream


def test_read_pstream_csv(tmp_path):
    data = "timestamp,pressure\n0.0,1.0\n1.0,2.0\n"
    file = tmp_path / "sample.csv"
    file.write_text(data)
    records = list(read_pstream(file))
    assert len(records) == 2
    assert records[0].pressure == 1.0
    assert records[0].timestamp == datetime.fromtimestamp(0.0, tz=timezone.utc)
    assert records[0].voltages == (0.0, 0.0, 0.0)
