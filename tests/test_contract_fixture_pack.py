from pathlib import Path
import zipfile


def test_contract_fixture_pack_exists_and_is_readable_from_zip() -> None:
    pack = Path(__file__).resolve().parents[1] / "echopress_contract_fixture_pack_v1.zip"
    assert pack.exists(), "Fixture pack zip must exist at repository root"

    with zipfile.ZipFile(pack) as archive:
        names = set(archive.namelist())

    required = {
        "contracts/pipeline_contract.yaml",
        "tests/fixtures/contracts/valid_pipeline_contract.yaml",
        "tests/fixtures/contracts/invalid_missing_required.yaml",
        "tests/fixtures/contracts/invalid_bad_enum.yaml",
        "tests/fixtures/datasets/smoke6/expected/expected_summary.json",
    }
    assert required.issubset(names)


def test_contract_fixture_pack_contains_expected_smoke6_raw_frames() -> None:
    pack = Path(__file__).resolve().parents[1] / "echopress_contract_fixture_pack_v1.zip"

    with zipfile.ZipFile(pack) as archive:
        raw_frames = [
            name
            for name in archive.namelist()
            if name.startswith("tests/fixtures/datasets/smoke6/raw/") and name.endswith(".npz")
        ]

    assert len(raw_frames) == 6
