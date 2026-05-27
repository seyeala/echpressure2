from __future__ import annotations

import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import numpy as np

from echopress.core.align_cleaner import AlignCleanerConfig, run_align_clean
from echopress.core.alignment_edit import revise_alignment_by_remove_list
from echopress.core.amplitude_filter import build_low_peak_remove_list
from echopress.core.mapping import align_streams
from echopress.core.tables import File2PressureMap, OscFiles, Signals, export_tables
from echopress.ingest import DatasetIndexer, load_ostream, read_pstream

from .state import PipelineFailure, PipelineStageRecord, build_artifact, load_pipeline_state, new_state, save_pipeline_state
from .validate import count_npz, validate_align_json, validate_index_json


class PipelineError(RuntimeError):
    pass


def _record_stage(state, record: PipelineStageRecord):
    state.stages[record.stage_name] = record
    save_pipeline_state(state)


def resolve_active_align(out_dir: Path, dataset_root: Optional[Path] = None) -> Dict[str, str]:
    dataset_root = dataset_root or out_dir
    state = load_pipeline_state(out_dir)
    candidates = []
    if state and 'active_align_json' in state.active_artifacts:
        candidates.append((Path(state.active_artifacts['active_align_json'].path), 'pipeline_state'))
    summary = out_dir / 'clean_align' / 'clean_align_summary.json'
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding='utf-8'))
            if 'output' in data:
                candidates.append((Path(data['output']), 'clean_align_summary'))
        except Exception:
            pass
    candidates += [
        (out_dir / 'clean_align' / 'align.cleaned.json', 'legacy_align_cleaned'),
        (out_dir / 'clean_align' / 'align.clean.json', 'canonical_align_clean'),
        (out_dir / 'align.filtered.json', 'filtered_align'),
        (out_dir / 'align.revised.json', 'revised_align'),
        (out_dir / 'align.json', 'raw_align'),
    ]
    for path, source in candidates:
        ok, _ = validate_align_json(path)
        if ok:
            return {'status': 'ok', 'active_align_path': str(path), 'source': source}
    return {'status': 'missing', 'active_align_path': '', 'source': 'none', 'message': 'No valid alignment artifact found'}


def run_prepare_align(dataset_root: Path, out_dir: Path, channel: int, baseline_samples: int, threshold_multiplier: float, alignment_error_max: float, mode: str = 'auto', force: bool = False, debug: bool = False) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_pipeline_state(out_dir) or new_state(dataset_root, out_dir)
    if not dataset_root.exists():
        raise PipelineError(f'dataset root missing: {dataset_root}')
    npz_count = count_npz(dataset_root)
    if npz_count == 0:
        raise PipelineError(f'No .npz files found under dataset_root: {dataset_root}')

    if mode == 'read-only':
        active = resolve_active_align(out_dir, dataset_root)
        return {'status': 'ready' if active['status'] == 'ok' else 'blocked', 'can_continue': active['status'] == 'ok', **active}

    index_path = out_dir / 'index.json'
    if force or not index_path.exists():
        rec = PipelineStageRecord(stage_name='index', status='running', started_at=state.updated_at)
        _record_stage(state, rec)
        t0 = perf_counter()
        idx = DatasetIndexer(dataset_root)
        sessions = idx.sessions()
        index_data = {
            'pstreams': {sid: [str(p) for p in idx.get_pstreams(sid, fallback=False)] for sid in sessions},
            'ostreams': {sid: [str(o) for o in idx.get_ostreams(sid, fallback=False)] for sid in sessions},
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index_data, indent=2), encoding='utf-8')
        rec.status = 'success'
        rec.outputs = {'index_json': str(index_path)}
        rec.finished_at = state.updated_at
        rec.duration_seconds = perf_counter() - t0
        _record_stage(state, rec)
    state.artifacts['index_json'] = build_artifact(out_dir, 'index_json', index_path, 'index')

    raw_align = out_dir / 'align.json'
    if force or not raw_align.exists():
        t0 = perf_counter(); rec = PipelineStageRecord(stage_name='align', status='running', started_at=state.updated_at); _record_stage(state, rec)
        indexer = DatasetIndexer(dataset_root)
        sessions = indexer.sessions()
        index_data = {
            'pstreams': {sid: [str(p) for p in indexer.get_pstreams(sid, fallback=False)] for sid in sessions},
            'ostreams': {sid: [str(o) for o in indexer.get_ostreams(sid, fallback=False)] for sid in sessions},
        }
        all_pstreams = [p for paths in index_data.get('pstreams', {}).values() for p in paths]
        signals = Signals(); osc_files = OscFiles(); fmap = File2PressureMap()
        for session, o_paths in sorted(index_data.get('ostreams', {}).items()):
            p_paths = index_data.get('pstreams', {}).get(session, []) or all_pstreams
            if not o_paths or not p_paths:
                continue
            o_path = Path(o_paths[0])
            if o_path.name in {'align.json', 'index.json'}:
                continue
            p_path = Path(p_paths[0])
            try:
                ostream = load_ostream(o_path)
                pstream = list(read_pstream(p_path))
                result = align_streams(ostream, pstream)
            except Exception as exc:
                msg = f'Failed to align session {session} (O-stream: {o_path}, P-stream: {p_path}): {exc}'
                rec.status = 'failed'; rec.error_message = msg; rec.duration_seconds = perf_counter() - t0; _record_stage(state, rec)
                if debug:
                    raise PipelineError(msg) from exc
                continue

            sid = ostream.session_id
            file_stamp = o_path.stem
            data = np.asarray(ostream.channels)
            if data.ndim == 2:
                data = data[:, 0] if data.shape[1] > 0 else np.array([])
            data = np.asarray(data).reshape(-1)
            if data.size == 0:
                osc_files.add(sid, file_stamp, 0, str(o_path))
            else:
                for idx, value in enumerate(data):
                    signals.add(sid, file_stamp, idx, float(value))
                    osc_files.add(sid, file_stamp, idx, str(o_path))
            if result.mapping >= 0:
                pressure_value = pstream[result.mapping].pressure
                fmap.add(sid, file_stamp, pressure_value, alignment_error=result.E_align)

        tables = export_tables(signals, osc_files, fmap, tall=True)
        raw_align.write_text(json.dumps(tables, default=float), encoding='utf-8')
        rec.status='success'; rec.outputs={'raw_align_json': str(raw_align)}; rec.duration_seconds=perf_counter()-t0; _record_stage(state, rec)
    ok, meta = validate_align_json(raw_align)
    if not ok:
        raise PipelineError(f'raw align invalid: {meta}')
    state.artifacts['raw_align_json'] = build_artifact(out_dir, 'raw_align_json', raw_align, 'raw_align', row_count=meta['row_count'], required_columns=meta['required_columns'], actual_columns=meta['actual_columns'])

    remove_list = out_dir / 'low_peak.remove.json'
    if force or mode in {'auto', 'resume'} and not remove_list.exists():
        t0 = perf_counter(); rec = PipelineStageRecord(stage_name='low_peak_filter', status='running', started_at=state.updated_at); _record_stage(state, rec)
        summary = build_low_peak_remove_list(align_table=raw_align, dataset_root=dataset_root, output_list=remove_list, channel=channel, baseline_samples=baseline_samples, baseline_seconds=None, threshold_multiplier=threshold_multiplier, include_missing=True)
        rec.status='success'; rec.outputs={'low_peak_remove_list': str(remove_list)}; rec.duration_seconds=perf_counter()-t0; _record_stage(state, rec)
    state.artifacts['low_peak_remove_list'] = build_artifact(out_dir, 'low_peak_remove_list', remove_list, 'low_peak_filter')

    filtered = out_dir / 'align.filtered.json'
    if force or mode in {'auto', 'resume'} and not filtered.exists():
        t0 = perf_counter(); rec = PipelineStageRecord(stage_name='revise_align', status='running', started_at=state.updated_at); _record_stage(state, rec)
        revise_alignment_by_remove_list(align_table=raw_align, remove_list=remove_list, output=filtered, match_key='path', invert=False)
        rec.status='success'; rec.outputs={'filtered_align_json': str(filtered)}; rec.duration_seconds=perf_counter()-t0; _record_stage(state, rec)
    ok, meta = validate_align_json(filtered)
    if ok:
        state.artifacts['filtered_align_json'] = build_artifact(out_dir, 'filtered_align_json', filtered, 'revise_align', row_count=meta['row_count'], required_columns=meta['required_columns'], actual_columns=meta['actual_columns'])

    clean_dir = out_dir / 'clean_align'
    canonical = clean_dir / 'align.clean.json'
    cleaned_legacy = clean_dir / 'align.cleaned.json'
    if force or mode in {'auto', 'resume'} and not cleaned_legacy.exists() and not canonical.exists():
        t0 = perf_counter(); rec = PipelineStageRecord(stage_name='clean_align', status='running', started_at=state.updated_at); _record_stage(state, rec)
        summary = run_align_clean(AlignCleanerConfig(align_table=filtered if filtered.exists() else raw_align, output_dir=clean_dir, alignment_error_max=alignment_error_max))
        if cleaned_legacy.exists() and not canonical.exists():
            canonical.write_text(cleaned_legacy.read_text(encoding='utf-8'), encoding='utf-8')
        rec.status='success'; rec.outputs={'clean_align_json': str(canonical if canonical.exists() else cleaned_legacy), 'clean_align_summary_json': str(clean_dir / 'clean_align_summary.json')}; rec.duration_seconds=perf_counter()-t0; _record_stage(state, rec)

    active = resolve_active_align(out_dir, dataset_root)
    if active['status'] != 'ok':
        raise PipelineError(active.get('message', 'No valid alignment artifact'))
    active_path = Path(active['active_align_path'])
    ok, meta = validate_align_json(active_path)
    state.artifacts['clean_align_json'] = build_artifact(out_dir, 'clean_align_json', active_path, 'clean_align', row_count=meta.get('row_count'), required_columns=meta.get('required_columns'), actual_columns=meta.get('actual_columns'))
    state.active_artifacts['active_align_json'] = state.artifacts['clean_align_json']
    save_pipeline_state(state)
    return {'status': 'ready', 'state_path': str((out_dir / '.echopress' / 'pipeline_state.json')), 'dataset_root': str(dataset_root), 'out_dir': str(out_dir), 'active_align_path': str(active_path), 'stages': {k: v.status for k, v in state.stages.items()}, 'can_continue': True, 'next_action': None}


def summarize_pipeline_state(out_dir: Path) -> Dict[str, object]:
    state = load_pipeline_state(out_dir)
    if not state:
        return {'status': 'missing', 'message': 'No local pipeline state or artifacts found. Rebuild required.'}
    missing = []
    artifacts = {}
    for k, v in state.artifacts.items():
        exists = Path(v.path).exists()
        status = 'OK' if exists else 'MISSING'
        artifacts[k] = {'path': v.path, 'status': status}
        if not exists:
            missing.append(k)
    active = state.active_artifacts.get('active_align_json')
    return {'status': 'ready' if not missing else 'blocked', 'active_align_path': active.path if active else None, 'stages': {k: s.status for k, s in state.stages.items()}, 'artifacts': artifacts, 'missing_artifacts': missing}
