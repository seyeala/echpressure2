# Architecture

Echpressure processes two unsynchronized data streams: a pressure stream (P-stream) providing timestamps and voltage triples, and an oscilloscope stream (O-stream) containing per-file waveforms sampled at a fixed interval. Each O-stream file is assigned a single scalar pressure label by aligning its midpoint time to the nearest P-stream timestamp, with uncertainty estimated from the local pressure derivative.

Alignment enforces a maximum allowable error ``O_max``. If the midpoint lies farther than this threshold from all P-stream timestamps, the file is rejected by default and marked in diagnostics. Setting ``reject_if_Ealign_gt_Omax=False`` retains the mapping but records the offending indices under ``E_align_violations``.

## Pipeline
1. **Ingestion & Indexing** – Resolve dataset paths, parse file and record metadata, and build in-memory tables for fast lookup. See [Dataset Indexer](dataset_indexer.md) for session lookup rules and pattern matching.
2. **Calibration & Mapping** – Convert voltages to calibrated pressure values and compute midpoint alignment, keeping track of error bounds.
3. **Adapters Layer 1** – Cycle-synchronous mappings such as PB-CSA, PLSTN, HMV, CEC, DTW-TA and MTP transform waveforms into fixed-length, shift-invariant representations.
4. **Adapters Layer 2** – Transforms applied to mapped cycles including Fourier spectrum, Hilbert-envelope, wavelet energies and MFCC features.
5. **Visualization** – Utilities to inspect raw, mapped and transformed data.
6. **Export** – On-demand routines generate NumPy-first datasets ready for machine-learning frameworks.

Hydra-driven configuration keeps the system modular while outputs remain framework-agnostic for future integration.
