# Troubleshooting

## Missing required files

If a stage fails with missing-file errors, verify prior-stage outputs exist:
- Macro: `first_peak_index.csv`, `global_window_size.json`
- Echo: `echo_peak_index.csv`, `echo_window_index.csv`
- Postprocess: `postprocessed_peak_windows.csv`

## No peaks detected

Try relaxing thresholds:
- Increase `--hilbert-frac`
- Lower `--min-prominence-rel`
- Lower `--min-height-rel`
- Reduce `--min-distance-samples`

## Invalid channel index

If you see channel-shape errors, confirm your `.npz` contains the requested channel and pass a valid `--channel`.

## T_global resolution errors

`detect-echo-peaks` needs `T_global_samples` from `global_window_size.json` or enough first-peak spacing to infer it. Re-run macro detection and inspect `first_peak_index.csv`.

## Non-deterministic dataset subsets

When debugging, constrain input size with macro options (for example `--max-files`) and keep outputs isolated per run directory.
