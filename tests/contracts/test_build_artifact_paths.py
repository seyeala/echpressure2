from pathlib import Path
from echopress.pipeline.state import build_artifact

def test_build_artifact_external_path_does_not_crash(tmp_path: Path):
    out_dir=tmp_path/'out'; out_dir.mkdir()
    external=tmp_path/'dataset'/'index.json'; external.parent.mkdir(); external.write_text('{}')
    a=build_artifact(out_dir,'index_json',external)
    assert a.artifact_scope=='external'
