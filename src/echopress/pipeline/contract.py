from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StageContract:
    stage_name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    success_checks: List[str] = field(default_factory=list)


@dataclass
class PipelineContract:
    stages: Dict[str, StageContract]


ALIGN_CONTRACT = PipelineContract(
    stages={
        "index": StageContract("index", ["dataset_root_exists"], ["index_json"], ["index_json_exists", "has_pstreams_ostreams"]),
        "raw_align": StageContract("raw_align", ["dataset_root_exists", "index_json"], ["raw_align_json"], ["row_count_gt_zero", "required_columns"]),
        "low_peak_filter": StageContract("low_peak_filter", ["raw_align_json", "dataset_root"], ["low_peak_remove_list"], ["output_exists"]),
        "revise_align": StageContract("revise_align", ["raw_align_json", "low_peak_remove_list"], ["filtered_align_json"], ["row_count_gt_zero", "required_columns"]),
        "clean_align": StageContract("clean_align", ["filtered_align_json_or_raw"], ["clean_align_json", "clean_align_summary_json", "active_align_json"], ["summary_output_exists", "row_count_gt_zero", "required_columns"]),
    }
)
