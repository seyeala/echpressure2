# Pipeline

This project provides a staged signal-processing pipeline through CLI commands:

1. `detect-macro-windows`
2. `detect-echo-peaks`
3. `postprocess-peak-windows`
4. `fft-postprocessed`

## Stage 1: detect-macro-windows

Inputs:
- Dataset root with `.npz` oscilloscope streams.
- Alignment table (`align.json`) mapping files to pressure values.

Primary outputs:
- `global_window_size.json`
- `first_peak_index.csv`
- `peak_to_peak_window_index.csv`

## Stage 2: detect-echo-peaks

Consumes macro outputs, especially `first_peak_index.csv` and `global_window_size.json`.

Primary outputs:
- `echo_peak_index.csv`
- `echo_window_index.csv`
- Optional cleaned windows (`cleaned_windows.npy`) when enabled.

## Stage 3: postprocess-peak-windows

Consumes:
- `echo_peak_index.csv`
- `echo_window_index.csv`

Produces:
- `secondary_peak_processed_waveforms.npy`
- `secondary_peak_processed_manifest.csv`
- `secondary_peak_processed_summary.json`

## Stage 4: fft-postprocessed

Consumes:
- `secondary_peak_processed_waveforms.npy`
- `secondary_peak_processed_manifest.csv`
- `secondary_peak_processed_summary.json`

Produces:
- `fft_cycles_per_window.npy`
- `fft_mag.npy`
- `fft_db.npy`
- `fft_relative_db.npy`
- `fft_manifest.csv`
- `fft_summary.json`

## Typical directory layout

- `macro/` for stage-1 outputs
- `echo/` for stage-2 outputs
- `post/` for stage-3 outputs
- `fft/` for stage-4 outputs

The pipeline is modular, so each stage can be re-run independently after parameter changes.
