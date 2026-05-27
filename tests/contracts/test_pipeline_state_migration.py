import json
from pathlib import Path

from typer.testing import CliRunner

from echopress.cli import app
from echopress.pipeline.runner import run_prepare_align
from echopress.pipeline.state import load_pipeline_state, state_path_for


def _write_old_state(out_dir: Path, artifact_path: Path, active_path: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "run_id": "old-run",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "dataset_root": str(out_dir),
        "out_dir": str(out_dir),
        "repo_commit": None,
        "package_version": None,
        "platform": "test",
        "python_version": "test",
        "config_hash": None,
        "dataset_fingerprint": None,
        "stages": {},
        "artifacts": {
            "index_json": {
                "logical_name": "index_json",
                "path": str(artifact_path),
            }
        },
        "active_artifacts": {},
        "failures": [],
        "history": [],
    }
    if active_path is not None:
        payload["active_artifacts"]["active_align_json"] = {
            "logical_name": "active_align_json",
            "path": str(active_path),
        }
    state_path_for(out_dir).parent.mkdir(parents=True, exist_ok=True)
    state_path_for(out_dir).write_text(json.dumps(payload), encoding="utf-8")


def test_old_state_without_artifact_scope_loads(tmp_path: Path):
    out_dir = tmp_path / "out"
    inside = out_dir / "index.json"
    inside.parent.mkdir(parents=True)
    inside.write_text("{}", encoding="utf-8")
    _write_old_state(out_dir, inside)

    state = load_pipeline_state(out_dir)

    assert state is not None
    assert state.artifacts["index_json"].artifact_scope == "pipeline"
    assert state.artifacts["index_json"].relative_path == "index.json"


def test_old_active_artifacts_without_artifact_scope_loads(tmp_path: Path):
    out_dir = tmp_path / "out"
    align = out_dir / "clean_align" / "align.clean.json"
    align.parent.mkdir(parents=True, exist_ok=True)
    align.write_text(json.dumps([{"path": "x", "pressure_value": 1.0}]), encoding="utf-8")
    _write_old_state(out_dir, out_dir / "index.json", active_path=align)

    state = load_pipeline_state(out_dir)

    assert state is not None
    assert state.active_artifacts["active_align_json"].artifact_scope == "pipeline"


def test_migrated_artifacts_get_scope_external_and_status(tmp_path: Path):
    out_dir = tmp_path / "out"
    external = tmp_path / "dataset" / "index.json"
    external.parent.mkdir(parents=True)
    external.write_text("{}", encoding="utf-8")
    _write_old_state(out_dir, external)

    state = load_pipeline_state(out_dir)

    art = state.artifacts["index_json"]
    assert art.artifact_scope == "external"
    assert art.relative_path is None
    assert art.exists is True
    assert art.status == "ok"


def test_prepare_align_after_restart_with_old_state_returns_ready(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "sample.npz").write_bytes(b"npz")

    out_dir = tmp_path / "out"
    valid_align = json.dumps([{"path": "x", "pressure_value": 1.0}])
    align = out_dir / "clean_align" / "align.clean.json"
    align.parent.mkdir(parents=True, exist_ok=True)
    align.write_text(valid_align, encoding="utf-8")
    _write_old_state(out_dir, out_dir / "index.json", active_path=align)

    result = run_prepare_align(dataset_root, out_dir, 0, 10, 10.0, 1.0, mode="read-only", force=False)

    assert result["can_continue"] is True
    assert result["active_align_path"] == str(align)


def test_prepare_align_json_returns_blocked_on_migration_failure(tmp_path: Path):
    out_dir = tmp_path / "out"
    state_path_for(out_dir).parent.mkdir(parents=True, exist_ok=True)
    state_path_for(out_dir).write_text("{not-json", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    runner = CliRunner()
    result = runner.invoke(app, [
        "prepare-align",
        "--dataset-root", str(dataset_root),
        "--out-dir", str(out_dir),
        "--json",
    ])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["can_continue"] is False
    assert payload["failed_stage"] == "load_pipeline_state"
    assert "next_action" in payload
