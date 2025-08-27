# echpressure2

Tools for processing and aligning pressure (`P`-stream) and oscilloscope (`O`-stream)
datasets. The project combines dataset indexing, calibration, alignment,
feature adapters and visualisation to produce machine–learning ready pressure
corpora. A broader roadmap is described in `plan.pdf`, and the data model is
formalised in `theory.pdf`.

## Project Overview

`echpressure2` ingests two unsynchronised data streams:

- **P-stream** – timestamped pressure measurements.
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
src/echopress/viz/  plotting helpers
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

# Run an adapter and save features
python -m echopress.cli adapt --adapter cec signal.npy -o features.npy
```

Existing commands such as `ingest`, `calibrate` and `viz` remain available.

## Configuration

Runtime configuration is managed with [Hydra](https://hydra.cc). The default
configuration in `conf/config.yaml` composes several YAML groups under `conf/`,
including:

* `dataset` – paths to example O- and P-streams (O-streams may be `.npz`, `.json`, or `.csv`)
* `calibration` – per-channel calibration coefficients
* `adapter` – parameters for signal adapters
* `viz` – options for plotting

By default, the library operates on channel `3`. Override this with
`calibration.channel` or the `ECHOPRESS_CHANNEL` environment variable.

The adapter section uses a nested schema. A minimal configuration looks like:

```yaml
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
python -m echopress.cli calibrate data.npy -o out.npy calibration.alpha=2.0
```

The `Settings` dataclass remains available for functions that expect it. In the
CLI, values from Hydra's `calibration` section are converted into `Settings`
instances for compatibility. `Settings.from_env` and
`echopress.config.load_settings` still allow configuration via environment
variables or explicit files when Hydra is not desired.
