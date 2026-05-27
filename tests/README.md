# Test Suite Organization

This repository uses `pytest` and keeps tests under the top-level `tests/` directory.

## Folder layout

- `tests/data/`
  - Put data integrity checks and schema validation tests here.
  - Examples: column/type checks, required-field checks, serialization/deserialization schema checks.

- `tests/contracts/`
  - Put CI/pipeline interface and artifact contract tests here.
  - Examples: checks that pipeline outputs include required files, stable field names, expected artifact metadata, and CLI/API contract behaviors relied on by downstream systems.

- `tests/fixtures/`
  - Put static sample inputs/outputs used by tests here.
  - Keep files small, deterministic, and safe to commit.

- `tests/conftest.py`
  - Shared `pytest` fixtures and test setup utilities used across test modules.

## Running tests locally

From the repository root:

- Run all tests:

  ```bash
  pytest
  ```

- Run a subset by folder:

  ```bash
  pytest tests/data tests/contracts
  ```

- Run a single test file:

  ```bash
  pytest tests/test_config.py
  ```

## CI expectations (every PR)

The following should run in CI on every pull request:

- Core automated test suite via `pytest`.
- Any contract tests in `tests/contracts/` that validate pipeline interfaces and artifact compatibility.
- Any data integrity/schema tests in `tests/data/` that protect expected data formats.

If CI time grows, prioritize keeping smoke/contract/data-integrity coverage on every PR and move heavier long-running jobs to scheduled workflows.
