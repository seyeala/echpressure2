from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

StageStatus = Literal["pending", "running", "success", "skipped", "stale", "failed", "missing", "superseded", "recovered"]
ArtifactStatus = Literal["ok", "missing", "stale", "superseded", "recovered"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineCheck:
    name: str
    status: str
    message: Optional[str] = None
    value: Any = None


@dataclass
class PipelineFailure:
    stage_name: str
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    log_path: Optional[str] = None
    when: str = field(default_factory=utcnow_iso)


@dataclass
class PipelineArtifact:
    logical_name: str
    path: str
    relative_path: Optional[str]
    exists: bool
    status: ArtifactStatus
    size_bytes: Optional[int] = None
    mtime: Optional[float] = None
    sha256: Optional[str] = None
    producer_stage: Optional[str] = None
    schema_status: Optional[str] = None
    row_count: Optional[int] = None
    required_columns: Optional[List[str]] = None
    actual_columns: Optional[List[str]] = None


@dataclass
class PipelineStageRecord:
    stage_name: str
    status: StageStatus
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    command: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    config_hash: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    checks: List[PipelineCheck] = field(default_factory=list)
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class PipelineState:
    schema_version: str
    run_id: str
    created_at: str
    updated_at: str
    dataset_root: str
    out_dir: str
    repo_commit: Optional[str]
    package_version: Optional[str]
    platform: str
    python_version: str
    config_hash: Optional[str]
    dataset_fingerprint: Optional[str]
    stages: Dict[str, PipelineStageRecord] = field(default_factory=dict)
    artifacts: Dict[str, PipelineArtifact] = field(default_factory=dict)
    active_artifacts: Dict[str, PipelineArtifact] = field(default_factory=dict)
    failures: List[PipelineFailure] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


SCHEMA_VERSION = "1.0"
STATE_RELATIVE_PATH = Path('.echopress') / 'pipeline_state.json'


def state_path_for(out_dir: Path) -> Path:
    return out_dir / STATE_RELATIVE_PATH


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(dataset_root: Path) -> str:
    npzs = sorted(dataset_root.rglob('*.npz'))
    digest = hashlib.sha256()
    digest.update(str(dataset_root).encode())
    digest.update(str(len(npzs)).encode())
    for p in npzs[:1000]:
        st = p.stat()
        digest.update(str(p.relative_to(dataset_root)).encode())
        digest.update(str(st.st_size).encode())
        digest.update(str(st.st_mtime_ns).encode())
    return digest.hexdigest()


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def new_state(dataset_root: Path, out_dir: Path, cfg: Optional[Dict[str, Any]] = None) -> PipelineState:
    cfg = cfg or {}
    now = utcnow_iso()
    return PipelineState(
        schema_version=SCHEMA_VERSION,
        run_id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        dataset_root=str(dataset_root),
        out_dir=str(out_dir),
        repo_commit=os.getenv('GIT_COMMIT'),
        package_version=None,
        platform=platform.platform(),
        python_version=sys.version,
        config_hash=config_hash(cfg),
        dataset_fingerprint=dataset_fingerprint(dataset_root) if dataset_root.exists() else None,
    )


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if hasattr(obj, '__dataclass_fields__'):
        data = asdict(obj)
        return {k: _to_dict(v) for k, v in data.items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def save_pipeline_state(state: PipelineState) -> Path:
    out_dir = Path(state.out_dir)
    path = state_path_for(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = utcnow_iso()
    payload = _to_dict(state)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return path


def load_pipeline_state(out_dir: Path) -> Optional[PipelineState]:
    path = state_path_for(out_dir)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding='utf-8'))
    stages = {k: PipelineStageRecord(**v) for k, v in raw.get('stages', {}).items()}
    artifacts = {k: PipelineArtifact(**v) for k, v in raw.get('artifacts', {}).items()}
    active = {k: PipelineArtifact(**v) for k, v in raw.get('active_artifacts', {}).items()}
    failures = [PipelineFailure(**v) for v in raw.get('failures', [])]
    raw['stages'] = stages
    raw['artifacts'] = artifacts
    raw['active_artifacts'] = active
    raw['failures'] = failures
    return PipelineState(**raw)


def build_artifact(out_dir: Path, logical_name: str, path: Path, producer_stage: Optional[str] = None, required_columns: Optional[List[str]] = None, actual_columns: Optional[List[str]] = None, row_count: Optional[int] = None, status: ArtifactStatus = 'ok') -> PipelineArtifact:
    exists = path.exists()
    st = path.stat() if exists else None
    resolved_out = out_dir.resolve()
    resolved_path = path.resolve() if exists else path
    relative_path: Optional[str]
    if exists and resolved_path.is_relative_to(resolved_out):
        relative_path = str(resolved_path.relative_to(resolved_out))
    elif exists:
        relative_path = None
    else:
        relative_path = str(path)
    return PipelineArtifact(
        logical_name=logical_name,
        path=str(resolved_path),
        relative_path=relative_path,
        exists=exists,
        status='ok' if exists and status == 'ok' else ('missing' if not exists else status),
        size_bytes=st.st_size if st else None,
        mtime=st.st_mtime if st else None,
        sha256=_hash_file(path) if exists and st and st.st_size < 50_000_000 else None,
        producer_stage=producer_stage,
        row_count=row_count,
        required_columns=required_columns,
        actual_columns=actual_columns,
    )
