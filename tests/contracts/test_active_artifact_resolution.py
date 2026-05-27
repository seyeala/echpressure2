import json
from pathlib import Path

from echopress.pipeline.runner import resolve_active_align


def _write_align(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"path":"x.npz","pressure_value":1.2}]), encoding='utf-8')


def test_resolver_prefers_cleaned(tmp_path: Path):
    out = tmp_path / 'out'; out.mkdir()
    _write_align(out / 'align.json')
    _write_align(out / 'align.filtered.json')
    _write_align(out / 'clean_align' / 'align.cleaned.json')
    r = resolve_active_align(out)
    assert r['status'] == 'ok'
    assert r['active_align_path'].endswith('clean_align/align.cleaned.json')
