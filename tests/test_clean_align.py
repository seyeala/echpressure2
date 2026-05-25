import json
from pathlib import Path

from echopress.core.macro_detector import MacroDetectorConfig, load_alignment_rows


def test_load_alignment_rows_resolves_nested_relative_paths(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    nested_existing = dataset_root / "session_a" / "run_01" / "file.npz"
    nested_existing.parent.mkdir(parents=True, exist_ok=True)
    nested_existing.write_bytes(b"npz")

    # Same basename at dataset root should not be used for nested relative path resolution.
    wrong_root_file = dataset_root / "file.npz"
    wrong_root_file.write_bytes(b"npz")

    missing_nested = "session_a/run_01/missing.npz"
    align_rows = [
        {"path": "session_a/run_01/file.npz", "pressure_value": 10.0},
        {"path": missing_nested, "pressure_value": 20.0},
    ]
    align_table = tmp_path / "align.json"
    align_table.write_text(json.dumps(align_rows))

    cfg = MacroDetectorConfig(
        dataset_root=dataset_root,
        align_table=align_table,
        output_dir=tmp_path / "out",
        npz_only=True,
    )

    rows = load_alignment_rows(cfg)

    assert len(rows) == 1
    assert rows.loc[0, "path"] == str(nested_existing.resolve())
    assert rows.loc[0, "path"] != str(wrong_root_file.resolve())
