from pathlib import Path

from echopress.pipeline.state import new_state, save_pipeline_state, load_pipeline_state, state_path_for


def test_pipeline_state_roundtrip(tmp_path: Path):
    dataset = tmp_path / 'ds'; dataset.mkdir()
    (dataset / 'a.npz').write_bytes(b'x')
    out = tmp_path / 'out'; out.mkdir()
    st = new_state(dataset, out)
    save_pipeline_state(st)
    p = state_path_for(out)
    assert p.exists()
    loaded = load_pipeline_state(out)
    assert loaded is not None
    assert loaded.schema_version
    assert loaded.dataset_root == str(dataset)
