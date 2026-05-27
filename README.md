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
  or `ai_log{ID}.csv` and contain `timestamp,pressure` columns.
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
Commands accept traditional `.pstream` text files or configured P-stream CSV
patterns such as `voltprsr*.csv` and `ai_log*.csv`:

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

### Revising alignment tables

Use the alignment-revision utilities to curate `align.json` rows before adapter
execution. `flag-low-peak` can create a removal list for files whose waveform
peak is not larger than a baseline absolute-amplitude average, and
`revise-align` writes the filtered alignment table for all adapters to reuse.

```bash
python -m echopress.cli flag-low-peak \
  --dataset-root /path/to/data \
  --align-table align.json \
  --output-list remove_low_peak.json \
  --baseline-samples 5000 \
  --threshold-multiplier 3

python -m echopress.cli revise-align \
  --align-table align.json \
  --remove-list remove_low_peak.json \
  --output align.revised.json \
  --match-key path

python -m echopress.cli adapt \
  --dataset-root /path/to/data \
  --align-table align.revised.json \
  --adapter cec \
  --output features.npy
```

Existing commands such as `ingest`, `calibrate` and `viz` remain available.

### RMCPE/TCIML pre-processing for marker-aware adaptation

The `adapt` command supports an optional two-stage preprocessing path enabled by
`--use-rmcpe-tciml`:

1. **Global period inference (RMCPE)** estimates one robust, cross-file window
   period `T_hat` from candidate incident peaks in all selected O-stream files.
2. **Local marker localization (TCIML)** then localizes per-file incident
   markers constrained by that global period and template-matching scores.

Conceptually, **RMCPE solves a dataset-level timing problem**, while **TCIML
solves a file-level localization problem**.

#### Configuration reference (major parameters)

The table below summarizes the core controls currently used by
`echopress.core.rmcpe.RMCPEConfig` and `echopress.core.tciml.TCIMLConfig`.

| Parameter | Default | Typical / valid range | Purpose |
| --- | --- | --- | --- |
| `RMCPE.T_min` | required | `>0` samples | Lower bound for candidate period search. |
| `RMCPE.T_max` | required | `>T_min` samples | Upper bound for candidate period search. |
| `RMCPE.raw_max_abs_min` | `0.0` | `>=0` | Rejects weak files before peak extraction. |
| `RMCPE.max_env_points` | `55000` | positive int | Envelope downsampling budget for block-max reduction. |
| `RMCPE.prominence` | `0.0` | `>=0` | `find_peaks` prominence threshold on envelope. |
| `RMCPE.distance` | `1` | int `>=1` | Minimum peak spacing in envelope points. |
| `RMCPE.width` | `None` | `None` or `>0` | Optional `find_peaks` width constraint. |
| `RMCPE.tau_T` | `1.0` | `>0` | Robust-loss transition/scaling control. |
| `RMCPE.lambda_` | `1.0` | `>0` | Exponential weighting strength for comb residual score. |
| `RMCPE.robust_loss` | `"huber"` | `huber` / `cauchy` / fallback L1 | Residual penalty family. |
| `RMCPE.bootstrap_count` | `200` | int `>=1` | Bootstrap draws for period CI. |
| `RMCPE.random_seed` | `0` | int | Reproducibility for bootstrap. |
| `RMCPE.poor_comb_score_min` | `1e-6` | `[0,1]` | Rejects files with poor comb consistency. |
| `TCIML.T_hat` | required | `>0` samples | Global period supplied to local marker search. |
| `TCIML.T_error_samples` | required | `>=0` | Period uncertainty used to size search radius. |
| `TCIML.peak_width_samples` | required | int `>=1` | Half-width of local marker/template window. |
| `TCIML.search_radius_min` | `8` | int `>=1` | Minimum search radius around expected centers. |
| `TCIML.alpha` | `0.75` | `[0,1]` | Blend weight between raw and envelope NCC scores. |
| `TCIML.C_min` | `0.25` | usually `[0,1]` | Minimum blended NCC score for acceptance. |
| `TCIML.P_min` | `0.0` | `>=0` | Minimum local prominence threshold. |
| `TCIML.W_minus` / `TCIML.W_plus` | `8` / `8` | int `>=0` | Pre/post context around localized marker. |
| `TCIML.envelope_rel_threshold` | `0.35` | `(0,1]` typical | Relative threshold for onset/end refinement. |
| `TCIML.template_peak_prominence` | `0.0` | `>=0` | Candidate-template peak screening. |

#### Output artifacts

When RMCPE/TCIML preprocessing is enabled, these artifacts are written in the
working directory:

* `window_period_summary.json` — global RMCPE result (`T_hat`, error estimate,
  bootstrap CI, acceptance counts, full config).
* `window_period_per_file.csv` — per-file RMCPE fit details (`T_i`, phase,
  robust score, rejection reason, peak counts).
* `incident_marker_table.csv` — TCIML marker-level table (expected vs matched
  centers, NCC scores, residuals, acceptance/reject reasons, window indices).
* Template `.npy` files:
  * `incident_template.npy` (raw template)
  * `incident_template_env.npy` (envelope template, when available)

#### Integration with cleanup and FTS

Accepted TCIML markers are used to crop each O-stream into incident windows
(`window_start_idx` → `window_end_idx`) before adapter extraction. The cropped
segments are concatenated and passed into the selected adapter pipeline,
including FTS. In practice, this acts as a **cleanup stage** that suppresses
non-incident regions and feeds cleaner, marker-aligned signal content to
frequency-domain transforms.

#### Troubleshooting

* **Insufficient incident candidates** (`insufficient_peaks` or template-build
  failures): lower `RMCPE.prominence`, reduce `RMCPE.distance`, widen
  `RMCPE.T_min/T_max`, or reduce `TCIML.template_peak_prominence`.
* **Harmonic ambiguity (`T` vs `2T`)**: tighten `RMCPE.T_min/T_max` around the
  expected physical period and verify accepted per-file `T_i` values in
  `window_period_per_file.csv`.
* **Poor comb match fraction** (`poor_comb_score`): relax
  `RMCPE.poor_comb_score_min` or increase signal quality before fitting.
* **Low NCC markers** (`score_low`): lower `TCIML.C_min`, retune
  `TCIML.alpha`, or increase template quality by filtering weak files.

#### Diagnostic note on largest-peak alignment

Largest-peak matching is useful as a diagnostic sanity check, but it is **not
the segmentation primitive** in this path. Segmentation windows are determined
by TCIML expected-center constraints plus NCC-based localization, not by a
single absolute largest-peak anchor.

### P-stream CSVs

Files like `voltprsr001.csv` and `ai_log001.csv` hold `timestamp,pressure`
pairs. The `DatasetIndexer` recognises the `voltprsr` and `ai_log` prefixes by
default and indexes the trailing identifier. See [docs/dataset_indexer.md](docs/dataset_indexer.md) for session handling, case-insensitive lookups and pattern matching. `read_pstream` loads these CSVs and yields
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
* `ingest` – patterns to recognise P-stream CSV files (default `['voltprsr', 'ai_log']`,
  matching names like `voltprsr*.csv` and `ai_log*.csv`)
* `mapping` – alignment and derivative parameters
* `calibration` – per-channel calibration coefficients
* `pressure` – which channel contains scalar pressure data; `pressure.scalar_channel`
  is zero-based, so the default `2` means physical channel 3
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

## DVC on Colab (AWS credentials)

When using DVC with AWS-backed remotes in Google Colab, set the following
environment variables before running DVC commands:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_DEFAULT_REGION`

## Pipeline contract and execution tracking

echopress now maintains a restart-safe pipeline ledger at:

- `<out_dir>/.echopress/pipeline_state.json` (canonical)

Use repo-owned orchestration commands instead of notebook path guessing:

```bash
python -m echopress.cli prepare-align --dataset-root /content/5Msagenerated --out-dir /content/drive/MyDrive/echpressure2_outputs/5Msagenerated --mode auto --json
python -m echopress.cli pipeline-status --out-dir /content/drive/MyDrive/echpressure2_outputs/5Msagenerated --json
```

The contract defines expected stage inputs/outputs/checks. The state ledger records what ran, what failed, produced artifacts, and the currently active alignment artifact (`active_align_path`).

Notebooks should call `prepare-align`/`pipeline-bootstrap` and consume returned JSON. Do not hardcode `align.json`, `align.filtered.json`, `align.cleaned.json`, or `align.clean.json` paths.
