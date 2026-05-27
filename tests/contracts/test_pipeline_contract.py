from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "pipeline_contract.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "contracts"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "metadata",
    "dataset",
    "allowed_stage_ids",
    "stages",
}

REQUIRED_STAGE_IDS = {
    "detect_macro_windows",
    "detect_echo_peaks",
    "postprocess_peak_windows",
    "fft_postprocessed",
}


def _load_yaml(path: Path) -> dict:
    assert path.exists(), f"Missing fixture file: {path}"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"Expected mapping YAML: {path}"
    return data


def _validate_contract(contract: dict) -> list[str]:
    errors: list[str] = []

    missing_top = REQUIRED_TOP_LEVEL_KEYS - set(contract.keys())
    if missing_top:
        errors.append(f"Missing required top-level keys: {sorted(missing_top)}")

    stage_ids = set(contract.get("allowed_stage_ids", []) or [])
    missing_stage_defs = REQUIRED_STAGE_IDS - stage_ids
    if missing_stage_defs:
        errors.append(f"Missing required stage ids in allowed_stage_ids: {sorted(missing_stage_defs)}")

    stages = contract.get("stages", []) or []
    stage_defs = {
        stage.get("stage_id"): stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage_id")
    }
    missing_stage_objects = REQUIRED_STAGE_IDS - set(stage_defs.keys())
    if missing_stage_objects:
        errors.append(f"Missing required stage definitions: {sorted(missing_stage_objects)}")

    if any(not isinstance(s, dict) for s in stages):
        errors.append("Every stage entry must be a mapping")

    for required_stage in REQUIRED_STAGE_IDS:
        stage = stage_defs.get(required_stage)
        if not stage:
            continue
        outputs = stage.get("required_outputs")
        if not isinstance(outputs, list) or not outputs:
            errors.append(
                f"Stage '{required_stage}' missing critical required_outputs list"
            )

    return errors


def test_pipeline_contract_yaml_has_required_keys_and_stages() -> None:
    contract = _load_yaml(CONTRACT)
    errors = _validate_contract(contract)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    ("fixture_name", "should_pass"),
    [
        ("valid_pipeline_contract.yaml", True),
        ("invalid_missing_required.yaml", False),
        ("invalid_bad_enum.yaml", False),
    ],
)
def test_contract_fixture_validation(fixture_name: str, should_pass: bool) -> None:
    fixture_contract = _load_yaml(FIXTURES / fixture_name)
    errors = _validate_contract(fixture_contract)

    if should_pass:
        assert not errors, f"Expected fixture {fixture_name} to pass; errors: {errors}"
    else:
        assert errors, f"Expected fixture {fixture_name} to fail"
