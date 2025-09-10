import pytest

from echopress.ingest import read_pstream, PStreamParseError


def test_read_pstream_unrecognised_line(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("foo\n")
    with pytest.raises(PStreamParseError) as excinfo:
        list(read_pstream(bad))
    msg = str(excinfo.value)
    assert f"{bad}:1:" in msg
    assert "Unrecognised P-stream line" in msg


def test_read_pstream_csv_bad_headers(tmp_path):
    data = "time,pressure\n0.0,1.0\n"
    file = tmp_path / "bad.csv"
    file.write_text(data)
    with pytest.raises(PStreamParseError) as excinfo:
        list(read_pstream(file))
    msg = str(excinfo.value)
    assert f"{file}:1:" in msg
    assert "Unrecognised timestamp" in msg
