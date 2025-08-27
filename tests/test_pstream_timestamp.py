from datetime import datetime, timezone

from echopress.ingest import parse_timestamp


def test_mdhmsu_format():
    result = parse_timestamp("M08-D19-H16-M24-S03-U.128")
    year = datetime.now(timezone.utc).year
    expected = datetime(year, 8, 19, 16, 24, 3, 128000, tzinfo=timezone.utc)
    assert result == expected
