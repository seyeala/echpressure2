# O-stream files

`load_ostream` converts raw O-stream files into the `OStream` dataclass. The
function defaults to *window mode*: each file describes a short capture window
with a start time derived from its filename and a fixed duration.

## Window-mode defaults

* **Duration** – the capture window spans `duration_s` seconds (default
  `0.02`). Two timestamps are produced: `[start, start + duration_s]`.
* **Filename timestamp** – when `use_filename_time` is true, the start time is
  parsed from the file stem using the pattern
  `M{month}-D{day}-H{hour}-M{minute}-S{second}-U.{micro}`. For example,
  `M08-D19-H16-M24-S03-U.128.os` resolves to a UTC start time of
  `2023-08-19T16:24:03.128Z` when no `base_year` is supplied.

Window mode emits an empty `(2, 0)` channel matrix and sets alignment midpoint
to `start + duration_s/2`.

O-stream files that only contain timestamps (zero channels) are valid. The
``align`` command notes their window-mode processing and records their paths in
the exported table so mappings are preserved.

### Sample usage

```python
from echopress.ingest import load_ostream

# Window-mode using the timestamp embedded in the filename
win = load_ostream("M08-D19-H16-M24-S03-U.128.os")
print(win.timestamps)

# Window-mode with an explicit start time and custom duration
exp = load_ostream(
    "capture.os",
    duration_s=0.05,
    start_time=1692456243.5,
    use_filename_time=False,
)
print(exp.timestamps)
```

## Non-window fallbacks

Passing `window_mode=False` enables robust parsing for stored data:

* **NPZ** – arrays named `timestamps`, `channels`, and optional metadata
  fields. NPZ also accepts `mV` as a channel alias and `time_ns`/`dt_ns` as
  timestamp sources (nanosecond absolute time or per-sample delta,
  respectively).
* **JSON/NDJSON/TXT** – keys `timestamps`, `channels`, and any extra metadata.
* **CSV** – headered or headerless matrices. When `override_file_timestamps`
  is true (default), timestamps are synthesised from `start_time` and
  `sampling_dt` rather than taken from the file.

These aliases are mapped automatically by `load_ostream` when `channels` or
`timestamps` are missing or empty.
