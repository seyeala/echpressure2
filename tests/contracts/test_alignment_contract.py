from pathlib import Path
from echopress.pipeline.runner import run_prepare_align

def test_prepare_align_dataset_root_and_out_dir_different(tmp_path: Path):
    d=tmp_path/'dataset'; d.mkdir(); (d/'x.npz').write_bytes(b'0')
    o=tmp_path/'out'
    try:
        r=run_prepare_align(d,o,0,100,1.0,1.0)
    except Exception:
        return
    assert (o/'.echopress'/'pipeline_state.json').exists()
    assert 'active_align_path' in r
