# Algorithms Docs: RMCPE + TCIML

This folder contains the formal algorithm note for robust window-period estimation and marker localization.

## Purpose

- **RMCPE** (Robust Multi-file Comb Period Estimation): estimates a robust global period across multiple files.
- **TCIML** (Template-Constrained Incident Marker Localization): localizes incident markers per file using the global period and a robust template constraint.

In short: **RMCPE is global period inference**, while **TCIML is local marker
localization**. The former estimates a cross-file period prior; the latter uses
that prior to segment incident windows per file.

## Runtime artifacts

Running the RMCPE→TCIML flow emits:

- `window_period_summary.json`
- `window_period_per_file.csv`
- `incident_marker_table.csv`
- `incident_template.npy` and (optionally) `incident_template_env.npy`

These artifacts are consumed by the adaptation stage to keep only accepted
incident windows before downstream transforms (including FTS).

## Module locations

- `src/echopress/core/rmcpe.py`
- `src/echopress/core/tciml.py`
- CLI wiring: `src/echopress/cli.py`

## Regenerate PDF locally

From repository root:

```bash
make docs-algorithms
```

This compiles:

- Source: `docs/algorithms/robust_window_period_and_marker_detection.tex`
- Artifact: `docs/algorithms/robust_window_period_and_marker_detection.pdf`
