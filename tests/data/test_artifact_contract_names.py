from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "pipeline_contract.yaml"


def _load_contract() -> dict:
    assert CONTRACT.exists(), f"Missing pipeline contract: {CONTRACT}"
    with CONTRACT.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "pipeline_contract.yaml must be a mapping"
    return data


def _stage_required_outputs(contract: dict, stage_id: str) -> set[str]:
    for stage in contract.get("stages", []) or []:
        if isinstance(stage, dict) and stage.get("stage_id") == stage_id:
            outputs = stage.get("required_outputs", []) or []
            return set(outputs)
    raise AssertionError(f"Missing stage definition: {stage_id}")


def test_detect_echo_peaks_requires_echo_window_values() -> None:
    contract = _load_contract()
    outputs = _stage_required_outputs(contract, "detect_echo_peaks")
    assert "echo_window_values.npy" in outputs


def test_postprocess_peak_windows_required_outputs() -> None:
    contract = _load_contract()
    outputs = _stage_required_outputs(contract, "postprocess_peak_windows")
    expected = {
        "secondary_peak_processed_waveforms.npy",
        "secondary_peak_processed_manifest.csv",
        "secondary_peak_processed_summary.json",
        "secondary_peak_gain_table.csv",
        "plot_marker_table.csv",
    }
    missing = expected - outputs
    assert not missing, f"Missing postprocess outputs: {sorted(missing)}"


def test_fft_postprocessed_required_outputs() -> None:
    contract = _load_contract()
    outputs = _stage_required_outputs(contract, "fft_postprocessed")
    expected = {
        "fft_relative_db.npy",
        "fft_cycles_per_window.npy",
        "fft_manifest.csv",
        "fft_mag.npy",
        "fft_db.npy",
        "fft_summary.json",
    }
    missing = expected - outputs
    assert not missing, f"Missing FFT outputs: {sorted(missing)}"
