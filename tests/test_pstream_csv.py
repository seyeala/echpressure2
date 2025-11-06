from datetime import timezone
import math

from echopress.ingest import read_pstream


def test_read_pstream_csv_infers_pressure_column(tmp_path):
    data = (
        "timestamp,Dev1/ai1,Dev1/ai2,Dev1/ai3\n"
        "2025-09-18 17:40:08.364162,5.763361801020377,6.978262006246402,5.753119004375975\n"
        "2025-09-18 17:40:08.610582,4.255366924344551,6.919200243074519,5.400315289813049\n"
    )
    file = tmp_path / "ai_log.csv"
    file.write_text(data)

    records = list(read_pstream(file))

    assert len(records) == 2
    first = records[0]
    assert first.timestamp.tzinfo is timezone.utc
    assert first.timestamp.isoformat() == "2025-09-18T17:40:08.364162+00:00"
    assert math.isclose(first.pressure, 5.763361801020377)
    assert first.voltages == (
        6.978262006246402,
        5.753119004375975,
    )


def test_read_pstream_csv_simple_header(tmp_path):
    data = "timestamp,pressure\n0.0,1.0\n1.0,2.0\n"
    file = tmp_path / "sample.csv"
    file.write_text(data)

    records = list(read_pstream(file))

    assert len(records) == 2
    assert records[0].pressure == 1.0
    assert records[0].timestamp.isoformat() == "1970-01-01T00:00:00+00:00"
    assert records[0].voltages is None
