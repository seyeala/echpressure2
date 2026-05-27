from pathlib import Path
import json

from echopress.pipeline.runner import run_prepare_align
from echopress.pipeline.state import build_artifact, load_pipeline_state


def test_build_artifact_external_path_does_not_crash(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    external = tmp_path / "dataset" / "index.json"
    external.parent.mkdir()
    external.write_text("{}", encoding="utf-8")

    artifact = build_artifact(out_dir, "index_json", external, "index")

    assert artifact.exists is True
    assert artifact.relative_path is None


def test_prepare_align_writes_index_and_state_under_out_dir(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "sample.npz").write_bytes(b"npz")
    (dataset_root / "sample.p").write_text("p", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    valid_align = json.dumps([{"path": "x", "pressure_value": 1.0}])
    (out_dir / "align.json").write_text(valid_align, encoding="utf-8")
    (out_dir / "low_peak.remove.json").write_text("[]", encoding="utf-8")
    (out_dir / "align.filtered.json").write_text(valid_align, encoding="utf-8")
    clean_dir = out_dir / "clean_align"
    clean_dir.mkdir()
    (clean_dir / "align.clean.json").write_text(valid_align, encoding="utf-8")

    result = run_prepare_align(
        dataset_root=dataset_root,
        out_dir=out_dir,
        channel=0,
        baseline_samples=10,
        threshold_multiplier=10.0,
        alignment_error_max=1.0,
        mode="auto",
        force=False,
    )

    assert isinstance(result, dict)
    assert "status" in result
    assert (out_dir / ".echopress" / "pipeline_state.json").exists()
    assert (out_dir / "index.json").exists()
    assert not (dataset_root / "index.json").exists()

    state = load_pipeline_state(out_dir)
    assert state is not None
    idx = state.artifacts["index_json"]
    assert Path(idx.path) == (out_dir / "index.json").resolve()
    assert idx.relative_path == "index.json"
