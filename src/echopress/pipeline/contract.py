from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ArtifactContract:
    logical_name: str
    relative_path_template: str | None
    required: bool = True
    kind: str = "file"
    required_columns: list[str] = field(default_factory=list)
    required_keys: list[str] = field(default_factory=list)
    min_rows: int | None = None
    allow_empty: bool = False

@dataclass
class StageContract:
    stage_name: str
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[ArtifactContract] = field(default_factory=list)
    success_checks: list[str] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    stale_policy: str = "config_hash_or_missing_outputs"

@dataclass
class PipelineContract:
    stages: dict[str, StageContract]

ALIGN_CONTRACT=StageContract(stage_name='align',outputs=[ArtifactContract('index_json','index.json'),ArtifactContract('clean_align_json','clean_align/align.clean.json')])
MACRO_CONTRACT=StageContract(stage_name='macro',depends_on=['align'],outputs=[ArtifactContract('macro_window_table_csv','{macro_dir}/macro_window_table.csv',kind='csv')])
ECHO_CONTRACT=StageContract(stage_name='echo',depends_on=['macro'],outputs=[ArtifactContract('echo_peak_index_csv','{echo_dir}/echo_peak_index.csv',kind='csv')])
POSTPROCESS_CONTRACT=StageContract(stage_name='postprocess',depends_on=['macro','echo'],outputs=[ArtifactContract('secondary_peak_processed_manifest_csv','{postprocess_dir}/secondary_peak_processed_manifest.csv',kind='csv')])
FFT_CONTRACT=StageContract(stage_name='fft',depends_on=['postprocess'],outputs=[ArtifactContract('fft_mag_npy','{fft_dir}/fft_mag.npy',kind='npy')])
PIPELINE_CONTRACT=PipelineContract(stages={s.stage_name:s for s in [ALIGN_CONTRACT,MACRO_CONTRACT,ECHO_CONTRACT,POSTPROCESS_CONTRACT,FFT_CONTRACT]})
