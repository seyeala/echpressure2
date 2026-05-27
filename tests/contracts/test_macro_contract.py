from pathlib import Path
from echopress.pipeline import runner

def test_prepare_macro_contract_records_artifacts(tmp_path: Path, monkeypatch):
    d=tmp_path/'d'; d.mkdir(); o=tmp_path/'o'; o.mkdir(parents=True)
    a=o/'clean_align'; a.mkdir(); (a/'align.clean.json').write_text('[]')
    monkeypatch.setattr(runner,'resolve_active_align',lambda *_:{'status':'ok','active_align_path':str(a/'align.clean.json')})
    def fake(cfg):
        cfg.output_dir.mkdir(parents=True,exist_ok=True)
        for n in ['macro_window_table.csv','first_peak_index.csv','global_window_size.json','peak_to_peak_window_index.csv']:
            (cfg.output_dir/n).write_text('x')
        return {}
    monkeypatch.setattr(runner,'run_macro_detection',fake)
    r=runner.run_prepare_macro(d,o)
    assert r['stage']=='macro'
