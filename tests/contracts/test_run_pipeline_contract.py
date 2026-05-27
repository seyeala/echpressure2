from pathlib import Path
from echopress.pipeline import runner

def test_run_pipeline_runs_all_requested_stages(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner,'run_prepare_align',lambda *a,**k:{'status':'ready'})
    monkeypatch.setattr(runner,'run_prepare_macro',lambda *a,**k:{'status':'ready'})
    monkeypatch.setattr(runner,'run_prepare_echo',lambda *a,**k:{'status':'ready'})
    monkeypatch.setattr(runner,'run_prepare_postprocess',lambda *a,**k:{'status':'ready'})
    monkeypatch.setattr(runner,'run_prepare_fft',lambda *a,**k:{'status':'ready'})
    r=runner.run_pipeline_full(tmp_path,tmp_path,['align','macro','echo','postprocess','fft'])
    assert set(r['stages'].keys())=={'align','macro','echo','postprocess','fft'}
