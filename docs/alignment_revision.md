# Alignment Revision and Waveform-Based Rejection

After `align`, the project writes an alignment table, usually `align.json`.
This table can be edited before running `adapt`.

## Removing listed datapoints

Use `revise-align` to remove rows from an alignment table.

```bash
python -m echopress.cli revise-align \
  --align-table align.json \
  --remove-list remove_list.json \
  --output align.revised.json \
  --match-key path
```

Supported match keys:

* `path`
* `path_basename`
* `file_stamp`
* `sid`
* `sid_file_stamp`
* `row_index`

The remove list may be JSON, TXT, or CSV.

## Building a remove list from waveform amplitude

Use `flag-low-peak` to scan files in `align.json` and flag files whose
maximum absolute amplitude is not larger than a baseline absolute-amplitude
average.

The rule is:

```text
max(abs(signal)) <= threshold_multiplier * mean(abs(signal[:x]))
```

where `x` is set by either `--baseline-samples` or `--baseline-seconds`.

```bash
python -m echopress.cli flag-low-peak \
  --dataset-root /path/to/data \
  --align-table align.json \
  --output-list remove_low_peak.json \
  --baseline-samples 5000 \
  --threshold-multiplier 3
```

Then revise the alignment:

```bash
python -m echopress.cli revise-align \
  --align-table align.json \
  --remove-list remove_low_peak.json \
  --output align.revised.json \
  --match-key path
```

Finally use the revised table:

```bash
python -m echopress.cli adapt \
  --dataset-root /path/to/data \
  --align-table align.revised.json \
  --adapter cec \
  --output features.npy
```
