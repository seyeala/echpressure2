from __future__ import annotations

import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import numpy as np

from echopress.core.align_cleaner import AlignCleanerConfig, run_align_clean
from echopress.core.macro_detector import MacroDetectorConfig, run_macro_detection
from echopress.core.echo_peaks import EchoPeakConfig, run_echo_peak_detection
from echopress.core.peak_window_postprocess import PeakWindowPostprocessConfig, run_peak_window_postprocess
from echopress.core.fft_export import FFTExportConfig, run_fft_postprocessed
from echopress.core.alignment_edit import revise_alignment_by_remove_list
from echopress.core.amplitude_filter import build_low_peak_remove_list
from echopress.core.mapping import align_streams
from echopress.core.tables import File2PressureMap, OscFiles, Signals, export_tables
from echopress.ingest import DatasetIndexer, load_ostream, read_pstream

from .state import PipelineFailure, PipelineStageRecord, build_artifact, load_pipeline_state, new_state, save_pipeline_state, state_path_for
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


def _state_or_new(dataset_root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    return load_pipeline_state(out_dir) or new_state(dataset_root, out_dir)

def _stage_result(stage: str, state, d: dict):
    return {"status":"ready","can_continue":True,"stage":stage,"state_path":str((Path(state.out_dir)/'.echopress'/'pipeline_state.json')), **d}

def run_prepare_macro(dataset_root: Path,out_dir: Path,align_table: Optional[Path]=None,run_mode: str='smoke',smoke_max_files: Optional[int]=5,channel:int=0,k_min:int=2,k_max:int=5,force_k: Optional[int]=None,block_size:int=10000,first_peak_search_frac:float=0.40,raw_max_abs_min:float=100.0,alignment_error_max:float=1.0,backward_full_windows: bool=True,progress_every:int=25,mode:str='auto',force:bool=False,debug:bool=False)->dict[str,object]:
    state=_state_or_new(dataset_root,out_dir)
    active=resolve_active_align(out_dir,dataset_root)
    if align_table is None:
        if active.get('status')!='ok':
            raise PipelineError('macro requires active align')
        align_table=Path(active['active_align_path'])
    macro_dir=out_dir / (f'macro_windows_SMOKE{smoke_max_files}' if run_mode=='smoke' else 'macro_windows_FULL')
    expected=[macro_dir/'macro_window_table.csv',macro_dir/'first_peak_index.csv',macro_dir/'global_window_size.json',macro_dir/'peak_to_peak_window_index.csv']
    exists=all(p.exists() for p in expected)
    reran=False; reused=False
    if mode in {'auto','resume','read-only'} and exists and not force:
        reused=True
    elif mode=='read-only' and not exists:
        raise PipelineError('macro artifacts missing in read-only mode')
    else:
        run_macro_detection(MacroDetectorConfig(dataset_root=dataset_root,align_table=align_table,output_dir=macro_dir,channel=channel,k_min=k_min,k_max=k_max,force_k=force_k,max_files=smoke_max_files if run_mode=='smoke' else None,block_size=block_size,first_peak_search_frac=first_peak_search_frac,raw_max_abs_min=raw_max_abs_min,max_alignment_error_s=alignment_error_max,backward_full_windows=backward_full_windows,progress_every=progress_every))
        reran=True
    state.artifacts['macro_window_table_csv']=build_artifact(out_dir,'macro_window_table_csv',macro_dir/'macro_window_table.csv','macro')
    state.artifacts['first_peak_index_csv']=build_artifact(out_dir,'first_peak_index_csv',macro_dir/'first_peak_index.csv','macro')
    state.artifacts['global_window_size_json']=build_artifact(out_dir,'global_window_size_json',macro_dir/'global_window_size.json','macro')
    state.artifacts['peak_to_peak_window_index_csv']=build_artifact(out_dir,'peak_to_peak_window_index_csv',macro_dir/'peak_to_peak_window_index.csv','macro')
    state.active_artifacts['active_macro_dir']=build_artifact(out_dir,'active_macro_dir',macro_dir,'macro')
    state.stages['macro']=PipelineStageRecord(stage_name='macro',status='success')
    save_pipeline_state(state)
    return _stage_result('macro',state,{"macro_dir":str(macro_dir),"active_artifacts":{k:v.path for k,v in state.active_artifacts.items()},"reused":reused,"reran":reran})

def run_prepare_echo(dataset_root: Path,out_dir: Path,detection_dir: Optional[Path]=None,mode:str='auto',force:bool=False,**kwargs)->dict[str,object]:
    state=_state_or_new(dataset_root,out_dir)
    if detection_dir is None:
        a=state.active_artifacts.get('active_macro_dir')
        if not a: raise PipelineError('echo requires active macro dir')
        detection_dir=Path(a.path)
    echo_dir=detection_dir/'echo_peaks_aggressive'
    if force or not (echo_dir/'echo_peak_index.csv').exists() or mode not in {'read-only','auto','resume'}:
        pass
    if not (echo_dir/'echo_peak_index.csv').exists() or force:
        run_echo_peak_detection(EchoPeakConfig(detection_dir=detection_dir,output_dir=echo_dir))
    state.artifacts['echo_peak_index_csv']=build_artifact(out_dir,'echo_peak_index_csv',echo_dir/'echo_peak_index.csv','echo')
    state.active_artifacts['active_echo_dir']=build_artifact(out_dir,'active_echo_dir',echo_dir,'echo')
    state.stages['echo']=PipelineStageRecord(stage_name='echo',status='success')
    save_pipeline_state(state)
    return _stage_result('echo',state,{"echo_dir":str(echo_dir)})

def run_prepare_postprocess(dataset_root: Path,out_dir: Path,macro_dir: Optional[Path]=None,echo_dir: Optional[Path]=None,mode:str='auto',force:bool=False,**kwargs)->dict[str,object]:
    state=_state_or_new(dataset_root,out_dir)
    if macro_dir is None: macro_dir=Path(state.active_artifacts['active_macro_dir'].path)
    if echo_dir is None: echo_dir=Path(state.active_artifacts['active_echo_dir'].path)
    post_dir=macro_dir/'post_peak_windows'
    if force or not (post_dir/'secondary_peak_processed_manifest.csv').exists():
        run_peak_window_postprocess(PeakWindowPostprocessConfig(macro_dir=macro_dir,echo_dir=echo_dir,output_dir=post_dir))
    state.active_artifacts['active_postprocess_dir']=build_artifact(out_dir,'active_postprocess_dir',post_dir,'postprocess')
    state.artifacts['secondary_peak_processed_manifest_csv']=build_artifact(out_dir,'secondary_peak_processed_manifest_csv',post_dir/'secondary_peak_processed_manifest.csv','postprocess')
    state.stages['postprocess']=PipelineStageRecord(stage_name='postprocess',status='success')
    save_pipeline_state(state)
    return _stage_result('postprocess',state,{"postprocess_dir":str(post_dir)})

def run_prepare_fft(dataset_root: Path,out_dir: Path,postprocess_dir: Optional[Path]=None,fft_bins:int=1024,mode:str='auto',force:bool=False,**kwargs)->dict[str,object]:
    state=_state_or_new(dataset_root,out_dir)
    if postprocess_dir is None: postprocess_dir=Path(state.active_artifacts['active_postprocess_dir'].path)
    fft_dir=postprocess_dir/'fft_outputs'
    if force or not (fft_dir/'fft_mag.npy').exists():
        run_fft_postprocessed(FFTExportConfig(postprocess_dir=postprocess_dir,output_dir=fft_dir,fft_bins=fft_bins))
    state.active_artifacts['active_fft_dir']=build_artifact(out_dir,'active_fft_dir',fft_dir,'fft')
    state.artifacts['fft_mag_npy']=build_artifact(out_dir,'fft_mag_npy',fft_dir/'fft_mag.npy','fft')
    state.stages['fft']=PipelineStageRecord(stage_name='fft',status='success')
    save_pipeline_state(state)
    return _stage_result('fft',state,{"fft_dir":str(fft_dir)})

def run_pipeline_full(dataset_root: Path,out_dir: Path,stages: list[str],**kwargs)->dict[str,object]:
    order=['align','macro','echo','postprocess','fft']
    selected=[s for s in order if s in stages]
    results={}
    for s in selected:
        if s=='align': results[s]=run_prepare_align(dataset_root,out_dir,channel=kwargs.get('channel',0),baseline_samples=kwargs.get('baseline_samples',10000),threshold_multiplier=kwargs.get('threshold_multiplier',50.0),alignment_error_max=kwargs.get('alignment_error_max',1.0),mode=kwargs.get('mode','auto'),force=kwargs.get('force',False))
        elif s=='macro': results[s]=run_prepare_macro(dataset_root,out_dir,run_mode=kwargs.get('run_mode','smoke'),smoke_max_files=kwargs.get('smoke_max_files',5),mode=kwargs.get('mode','auto'),force=kwargs.get('force',False))
        elif s=='echo': results[s]=run_prepare_echo(dataset_root,out_dir,mode=kwargs.get('mode','auto'),force=kwargs.get('force',False))
        elif s=='postprocess': results[s]=run_prepare_postprocess(dataset_root,out_dir,mode=kwargs.get('mode','auto'),force=kwargs.get('force',False))
        elif s=='fft': results[s]=run_prepare_fft(dataset_root,out_dir,fft_bins=kwargs.get('fft_bins',1024),mode=kwargs.get('mode','auto'),force=kwargs.get('force',False))
    st=load_pipeline_state(out_dir)
    return {"status":"ready","can_continue":True,"state_path":str(state_path_for(out_dir)),"selected_stages":selected,"stages":results,"active_artifacts":{k:v.path for k,v in (st.active_artifacts.items() if st else [])}}
