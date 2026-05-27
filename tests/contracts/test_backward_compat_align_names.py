import json
from pathlib import Path

from echopress.pipeline.runner import resolve_active_align


def _w(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"path":"a","pressure_value":1.0}]), encoding='utf-8')


def test_resolver_accepts_align_clean_name(tmp_path: Path):
    out = tmp_path / 'out'; out.mkdir()
    _w(out / 'clean_align' / 'align.clean.json')
    r = resolve_active_align(out)
    assert r['status'] == 'ok'
    assert r['active_align_path'].endswith('align.clean.json')
