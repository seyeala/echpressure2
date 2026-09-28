# Architecture

Echpressure processes two unsynchronized data streams: a P-stream of timestamped
numeric measurements and an O-stream of oscilloscope waveforms sampled at a
fixed interval. P-stream values may be raw voltage or calibrated pressure;
see [P-stream formats and calibration](pstream.md). Alignment assigns a scalar
label from the nearest P-stream timestamp to an O-stream file's midpoint,
with uncertainty estimated from the local derivative. Labels and their
uncertainties have pressure units only when the input values have been
calibrated to those units.

Alignment enforces a maximum allowable error ``O_max``. If the midpoint lies farther than this threshold from all P-stream timestamps, the file is rejected by default and marked in diagnostics. Setting ``reject_if_Ealign_gt_Omax=False`` retains the mapping but records the offending indices under ``E_align_violations``.

## Pipeline
1. **Ingestion & Indexing** – Resolve dataset paths, parse file and record metadata, and build in-memory tables for fast lookup. See [Dataset Indexer](dataset_indexer.md) for session lookup rules and pattern matching.
2. **Calibration & Mapping** – Explicitly select and calibrate raw sensor values before using them as physical pressure labels, then compute midpoint alignment and error bounds. Reading and aligning P-streams do not automatically apply the calibration helpers.
2.5. **Alignment Revision / Quality Filtering** – After file-level midpoint alignment, users may remove poor-quality rows using alignment metadata or waveform-derived metrics. The revised alignment table is then passed to `adapt --align-table`.
3. **Adapters Layer 1** – Segment or map waveforms into cycle representations according to the selected adapter. Capabilities depend on the implementation: [DTW-TA](adapters/dtw_ta.md) currently performs basic segmentation, while DTW template alignment and averaging are planned. Fixed-length output alone does not establish shift invariance.
4. **Adapters Layer 2** – Transform mapped cycles using Fourier spectra, [Hilbert envelopes](adapters/hte.md), wavelet energies, or the [simplified cepstral approximation](adapters/mfcc.md) exposed as `mfcc`. The latter has no mel filter bank. Hilbert envelopes shift with their input and need alignment or appropriate aggregation for shift-invariant features.
5. **Visualization** – Utilities to inspect raw, mapped and transformed data.
6. **Export** – On-demand routines generate NumPy-first datasets ready for machine-learning frameworks.

Configuration is handled by Typer + Pydantic Settings: commands share a single
validated `Settings` instance loaded from YAML/JSON files, environment
variables, or inline overrides. This keeps the system modular while outputs
remain framework-agnostic for future integration.
