# Configuration Reference

Configuration can be provided via `--config` YAML/JSON files and overridden with CLI options.

## Macro detection (`detect-macro-windows`)

Important options include:
- `--dataset-root`
- `--align-table`
- `--output-dir`
- `--channel`
- `--k-min`, `--k-max`, `--force-k`
- `--first-peak-search-frac`
- `--snap-tol-frac`
- `--write-signatures`

## Echo detection (`detect-echo-peaks`)

Important options include:
- `--detection-dir`
- `--output-dir`
- `--use-registered/--all-first-peaks`
- `--hilbert-frac`
- `--min-prominence-rel`
- `--min-height-rel`
- `--min-distance-samples`
- `--refine-radius-samples`
- `--max-peaks-per-window`

## Postprocess (`postprocess-peak-windows`)

Important options include:
- `--echo-dir`
- `--output-dir`
- `--max-echo-peak-order`

## FFT export (`fft-postprocessed`)

Important options include:
- `--postprocess-dir`
- `--output-dir`
- `--fft-bins`

## Global configuration

The CLI also supports top-level overrides:
- `--config path/to/config.yaml`
- `--set dotted.path=value`
- `--dataset-root`
- `--adapter-name`
- `--align-table`
