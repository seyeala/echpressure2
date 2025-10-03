# echpressure2

Tools for processing and aligning pressure (`P`-stream) and oscilloscope (`O`-stream)
datasets. The project combines dataset indexing, calibration, alignment,
feature adapters and visualisation to produce machine–learning ready pressure
corpora. A broader roadmap is described in `plan.pdf`, and the data model is
formalised in `theory.pdf`.

## Project Overview

`echpressure2` ingests two unsynchronised data streams:

 - **P-stream** – timestamped pressure measurements expressed in millimetres of
  mercury (mmHg). P-stream CSVs are conventionally named `voltprsr{ID}.csv`
  and contain `timestamp,pressure` columns.
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
conf/       legacy Hydra presets (reference only)
config/     example Pydantic settings files
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

Once the CLI exposes the full pipeline, a typical flow looks like below.
Commands accept traditional `.pstream` text files or `voltprsr*.csv`
P-streams:

```bash
# Build dataset indices from configured paths (supports .pstream and CSV)
python -m echopress.cli index

# Align O-stream files to the P-stream using the cached index
python -m echopress.cli align /data --window-mode --duration 0.05 --base-year 2023

# Run an adapter on files within a pressure range and save features
python -m echopress.cli adapt --adapter cec --pr-min 80 --pr-max 120 --n 5 --output features.npy
```

The `index` command writes an `index.json` file under the dataset root. The
`align` step consumes this digest, aligns the first O-/P-stream pair in each
session and emits a consolidated `align.json` table.  When ``--window-mode`` is
used, O-stream files are treated as timestamped capture windows; even files with
no channel data are still recorded so their paths appear in the exported table.
Downstream utilities like `adapt` read this table to locate files by pressure
value.

Existing commands such as `ingest`, `calibrate` and `viz` remain available.

### P-stream CSVs

Files like `voltprsr001.csv` hold `timestamp,pressure` pairs. The
`DatasetIndexer` recognises the `voltprsr` prefix by default and indexes the
trailing identifier. See [docs/dataset_indexer.md](docs/dataset_indexer.md) for session handling, case-insensitive lookups and pattern matching. `read_pstream` loads these CSVs and yields
`PStreamRecord` objects with parsed timestamps and floating-point pressures.

```python
from echopress.ingest import DatasetIndexer, read_pstream

# Find and read the first P-stream with ID "001"
indexer = DatasetIndexer("/data")
pstream_path = indexer.first_pstream("001")
for record in read_pstream(pstream_path):
    print(record.timestamp, record.pressure)
```

## Configuration

Runtime configuration is managed with [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
The root model lives in `echopress.config.Settings` and mirrors the legacy Hydra
defaults. A ready-to-use template is shipped in `config/example.yaml`.

Settings can be supplied in three complementary ways:

1. **Configuration file** – pass `--config path/to/settings.yaml` to the CLI.
   Files may be JSON or YAML; nested keys must match the structure described
   below. The provided template includes sensible defaults for calibration,
   mapping and adapter behaviour.
2. **Environment variables** – any field can be overridden by defining
   `ECHOPRESS_<SECTION>__<FIELD>` (note the double underscore to separate nested
   keys). For example,
   `export ECHOPRESS_DATASET__ROOT=/data/echopress` changes the dataset root for
   subsequent commands.
3. **Inline overrides** – use the `--set` option to update dotted keys without
   editing files, e.g. `--set mapping.O_max=0.005 --set adapter.n=4`.

Key sections available in the settings schema include:

* `dataset` – dataset root directory and timestamp metadata
* `ingest` – patterns to recognise P-stream CSV files (default `['voltprsr']`,
  matching names like `voltprsr*.csv`)
* `mapping` – alignment and derivative parameters
* `calibration` – per-channel calibration coefficients
* `pressure` – which channel contains scalar pressure data
* `units` – display units for pressure and voltage
* `timestamp` – parsing controls
* `quality` – quality gates for downstream processing
* `adapter` – parameters for signal adapters
* `viz` – options for plotting

Loading the example configuration and overriding individual values looks like:

```bash
# Use the example settings and override the dataset root on the fly
python -m echopress.cli --config config/example.yaml --set dataset.root=/data index

# Environment variables are also respected
export ECHOPRESS_ADAPTER__NAME=cec
python -m echopress.cli --config config/example.yaml adapt --n 8 --output features.npy
```

The `Settings` container inherits from Pydantic's `BaseSettings`, so creating an
instance (`Settings()`) automatically applies environment overrides. Use
`echopress.config.load_settings` to load JSON/YAML files programmatically.
