# Algorithms Docs: RMCPE + TCIML

This folder contains the formal algorithm note for robust window-period estimation and marker localization.

## Purpose

- **RMCPE** (Robust Multi-file Comb Period Estimation): estimates a robust global period across multiple files.
- **TCIML** (Template-Constrained Incident Marker Localization): localizes incident markers per file using the global period and a robust template constraint.

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
