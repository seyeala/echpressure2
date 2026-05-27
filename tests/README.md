# Test Layout

This directory contains all automated tests for the repository.

## Subfolders

- `tests/data/`
  - Data integrity and schema validation tests.
  - Use for checks like required columns, value ranges, null constraints, and file format compatibility.

- `tests/contracts/`
  - CI/pipeline interface and artifact contract tests.
  - Use for tests that validate stable boundaries between stages (CLI inputs/outputs, generated artifact names, expected metadata, serialization contracts).

- `tests/fixtures/`
  - Static sample input/output files used by tests.
  - Store small, deterministic fixtures only (CSV/NPZ/JSON/YAML snippets, expected outputs, golden files).

## Shared fixtures

This is a Python repository; shared pytest fixtures should live in `tests/conftest.py` so they are available across test modules.

## Running tests locally

From the repository root:

```bash
pytest
```

Or, matching the project pytest defaults (`-q` from `pyproject.toml`):

```bash
pytest -q
```

You can also run a subset, for example:

```bash
pytest tests/data
pytest tests/contracts
pytest tests/test_cli_commands.py
```

## CI expectations for pull requests

On every PR, CI is expected to run the full pytest suite under `tests/`, including:

- Existing module-level tests (for core behavior and regressions).
- `tests/data/` data/schema integrity tests.
- `tests/contracts/` interface/artifact contract tests.

Tests that rely on large external datasets or environment-specific resources should be marked and configured separately, but should not replace the required PR gate tests above.
