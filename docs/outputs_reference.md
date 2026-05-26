# Outputs Reference

## Macro stage outputs

- `global_window_size.json`: includes `T_global_samples` and windowing summary fields.
- `first_peak_index.csv`: one row per detected first peak per macro window.
- `first_peak_index.registered.csv`: registered subset/annotations for backward common windows.
- `peak_to_peak_window_index.csv`: windows built between adjacent first peaks.

## Echo stage outputs

- `echo_peak_index.csv`: detected secondary echo peaks (can be multiple per window).
- `echo_window_index.csv`: per-window bookkeeping and detection status.
- `echo_peak_summary.json`: stage summary counts.
- Optional: `cleaned_windows.npy`, `cleaned_window_index.csv`.

## Postprocess stage outputs

- `secondary_peak_processed_waveforms.npy`: processed waveform matrix with shape `[n_files, n_samples]`.
- `secondary_peak_processed_manifest.csv`: per-row metadata aligned to the waveform matrix.
- `secondary_peak_processed_summary.json`: summary counts.

## FFT stage outputs

- `fft_cycles_per_window.npy`: frequency axis for FFT bins (cycles per processed window).
- `fft_mag.npy`: FFT magnitude matrix.
- `fft_db.npy`: FFT magnitude in dB.
- `fft_relative_db.npy`: row-normalized dB features for ML.
- `fft_manifest.csv`: per-row metadata aligned to FFT matrices.
- `fft_summary.json`: summary metadata.
