from echopress.ingest import read_pstream


def test_read_pstream_csv(tmp_path):
    csv_path = tmp_path / "VoltPrsr001.csv"
    csv_path.write_text("\n".join([
        "timestamp,pressure",
        "0.0,1.0",
        "1.0,2.0",
    ]))
    records = list(read_pstream(csv_path))
    assert [r.pressure for r in records] == [1.0, 2.0]
    assert [r.timestamp.timestamp() for r in records] == [0.0, 1.0]
    assert all(r.voltages is None for r in records)
