# echpressure2

Tools for processing and aligning pressure (`P`-stream) and oscilloscope (`O`-stream)
datasets. The project combines dataset indexing, calibration, alignment,
feature adapters and visualisation to produce machine–learning ready pressure
corpora. A broader roadmap is described in `plan.pdf`, and the data model is
formalised in `theory.pdf`.

## Project Overview

`echpressure2` ingests two unsynchronised data streams:

- **P-stream** – timestamped pressure measurements expressed in millimetres of
  mercury (mmHg).
- **O-stream** – oscilloscope files containing uniformly sampled waveforms.

Each O-stream file is mapped to the nearest P-stream timestamp, calibrated and
transformed by modular adapters to yield features for analysis or downstream
learning tasks.

## Architecture

The codebase is organised into focused modules:

- **ingest** – parse raw streams and build in-memory registries.
- **core** – mapping, alignment and uncertainty bounds.
- **adapters** – cycle-synchronous feature extractors and transform layers.
- **export** – prepare NumPy-based datasets for other frameworks.
- **viz** – plotting utilities for quick inspection.

## Repository Layout

```
conf/       Hydra configuration groups
src/        library code under `echopress`
viz/        plotting helpers
tests/      unit tests
plan.pdf    project roadmap
theory.pdf  dataset theory
```

Key modules live in `src/echopress/`:

```
adapters/  feature adapters
core/      alignment and mapping logic
ingest/    stream loaders and indexers
export/    dataset export helpers
utils/ & types.py  shared utilities and data types
```

## Usage

Once the CLI exposes the full pipeline, a typical flow looks like:

```bash
# Build dataset indices from configured paths
python -m echopress.cli index

# Align O-stream files to the P-stream and compute uncertainty bounds
python -m echopress.cli align

# Run an adapter on files within a pressure range and save features
python -m echopress.cli adapt --adapter cec --pr-min 80 --pr-max 120 --n 5 --output features.npy
```

Existing commands such as `ingest`, `calibrate` and `viz` remain available.

## Configuration

Runtime configuration is managed with [Hydra](https://hydra.cc). The default
configuration in `conf/config.yaml` composes several YAML groups under `conf/`,
including:

* `dataset` – paths to example O- and P-streams (O-streams may be `.npz`, `.json`, or `.csv`)
* `mapping` – alignment and derivative parameters
* `calibration` – per-channel calibration coefficients
* `pressure` – which channel contains scalar pressure data
* `units` – display units for pressure and voltage
* `timestamp` – parsing controls
* `quality` – quality gates for downstream processing
* `adapter` – parameters for signal adapters
* `viz` – options for plotting

The configuration uses a nested schema. A minimal configuration looks like:

```yaml
calibration:
  alpha: [1.0, 1.0, 1.0]
  beta: [0.0, 0.0, 0.0]
pressure:
  scalar_channel: 2
units:
  pressure: Pa
  voltage: V
timestamp:
  format: null
  timezone: UTC
  year_fallback: 1970
quality:
  reject_if_Ealign_gt_Omax: true
  min_records_in_W: 3
adapter:
  name: cec
  output_length: 0  # use full output
  period_est:
    fs: 10.0
    f0: 2.0
```

Commands in `echopress.cli` are wrapped by `hydra.main` so overrides can be
passed directly on the command line. For example, to adjust calibration
parameters at runtime:

```bash
python -m echopress.cli calibrate data.npy -o out.npy calibration.alpha='[2.0]'
```

The `Settings` container remains available for functions that expect it. In the
CLI, values from Hydra's nested sections are converted into `Settings`
instances for compatibility. `Settings.from_env` and
`echopress.config.load_settings` still allow configuration via environment
variables or explicit files when Hydra is not desired.
