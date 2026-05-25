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

- `postprocessed_peak_windows.csv`: merged window-level features.
- `postprocess_peak_windows_summary.json`: summary counts.

## FFT stage outputs

- `postprocessed_fft.npy`: FFT magnitude vector.
- `postprocessed_fft.csv`: tabular FFT magnitudes per bin.
- `fft_postprocessed_summary.json`: summary metadata.
