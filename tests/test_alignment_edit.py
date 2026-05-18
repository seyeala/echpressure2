import json

from echopress.core.alignment_edit import revise_alignment_by_remove_list


def _write_alignment(path):
    rows = [
        {
            "path": "/a/file1.npz",
            "sid": "s1",
            "file_stamp": "f1",
            "pressure_value": 1.0,
        },
        {
            "path": "/b/file2.npz",
            "sid": "s2",
            "file_stamp": "f2",
            "pressure_value": 2.0,
        },
        {
            "path": "/c/file3.npz",
            "sid": "s1",
            "file_stamp": "f3",
            "pressure_value": 3.0,
        },
    ]
    path.write_text(json.dumps(rows))
    return rows


def _revise(tmp_path, remove_items, match_key, *, invert=False):
    align = tmp_path / "align.json"
    remove = tmp_path / "remove.json"
    out = tmp_path / "align.revised.json"
    _write_alignment(align)
    remove.write_text(json.dumps(remove_items))

    summary = revise_alignment_by_remove_list(
        align_table=align,
        remove_list=remove,
        output=out,
        match_key=match_key,
        invert=invert,
    )

    return json.loads(out.read_text()), summary


def test_revise_alignment_by_path(tmp_path):
    kept, summary = _revise(tmp_path, ["/a/file1.npz"], "path")

    assert [row["path"] for row in kept] == ["/b/file2.npz", "/c/file3.npz"]
    assert summary["removed_rows"] == 1
    assert summary["kept_rows"] == 2


def test_revise_alignment_by_path_basename(tmp_path):
    kept, summary = _revise(tmp_path, ["/old/location/file2.npz"], "path_basename")

    assert [row["path"] for row in kept] == ["/a/file1.npz", "/c/file3.npz"]
    assert summary["removed_rows"] == 1


def test_revise_alignment_by_file_stamp(tmp_path):
    kept, summary = _revise(tmp_path, [{"file_stamp": "f3"}], "file_stamp")

    assert [row["file_stamp"] for row in kept] == ["f1", "f2"]
    assert summary["removed_rows"] == 1


def test_revise_alignment_by_sid_file_stamp(tmp_path):
    kept, summary = _revise(
        tmp_path,
        [{"sid": "s1", "file_stamp": "f1"}],
        "sid_file_stamp",
    )

    assert [row["file_stamp"] for row in kept] == ["f2", "f3"]
    assert summary["removed_rows"] == 1


def test_revise_alignment_by_row_index(tmp_path):
    kept, summary = _revise(tmp_path, [1], "row_index")

    assert [row["path"] for row in kept] == ["/a/file1.npz", "/c/file3.npz"]
    assert summary["removed_rows"] == 1


def test_revise_alignment_invert_keeps_only_listed_rows(tmp_path):
    kept, summary = _revise(tmp_path, ["/b/file2.npz"], "path", invert=True)

    assert [row["path"] for row in kept] == ["/b/file2.npz"]
    assert summary["removed_rows"] == 2
    assert summary["kept_rows"] == 1
