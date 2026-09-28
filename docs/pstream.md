# P-stream files

P-streams contain timestamps and numeric measurements. Those measurements may
be raw sensor voltages or already calibrated pressure values. The parser's
field name `pressure` does not establish units or perform calibration.

Files are conventionally named `voltprsr{ID}.csv` or `ai_log{ID}.csv`, for
example `voltprsr001.csv` or `ai_log001.csv`, so `DatasetIndexer` can extract
the identifier. Runtime filename patterns are configured through
`Settings.ingest.pstream_csv_patterns`; the files under `conf/` are legacy
reference presets.

## Accepted layouts and column selection

| Layout | Example | Selected value |
| --- | --- | --- |
| Headered CSV opened by file path | `timestamp,pressure` | First non-timestamp header containing `pressure`, case-insensitively. |
| Headered CSV with channel names | `timestamp,Dev1/ai0,Dev1/ai1` | If no pressure header exists, the first non-timestamp column: `Dev1/ai0` here. |
| Paired-line text | Timestamp line followed by `v1,v2,v3` or space-separated values | The zero-based `value_col` argument; default `2` selects the third numeric value. |
| Simple text line | `2026-09-28T12:00:00Z,1.25` or a timestamp token followed by whitespace and a value | The value immediately after the timestamp token. |

For headered CSVs, the first header containing `timestamp` is the time
column (case-insensitive). The reader prefers a header containing
`pressure` over other measurement headers, even if that header is not the
first measurement column. Use unambiguous names and inspect the selected
column. Other non-empty measurement fields may be retained in
`PStreamRecord.voltages`; their name also does not validate physical units.

The `value_col` argument only controls paired-line text selection; it does
not override headered CSV selection. Likewise, `pressure.scalar_channel`
selects the calibration coefficient index in `apply_calibration`, not a CSV
header. Its default `2` means the third coefficient, not necessarily a
particular physical device channel. The current `align` and `prepare-align`
entry points call `read_pstream` with its default `value_col=2`; they do not
forward `pressure.scalar_channel` to the parser.

CSV header detection applies to a `.csv` file opened by path. File-like
objects use the text parser. Use a single timestamp token such as ISO 8601
with `T` for simple text lines; a date and time separated by a space should
be supplied in a headered CSV or on its own line in a paired-line record.

### Example: raw PhantomTest log

```csv
timestamp,Dev1/ai0,Dev1/ai1
2026-09-28T12:00:00Z,1.25,2.50
```

This record parses with `pressure=1.25` because `Dev1/ai0` is the first
measurement column. If the measurements are volts, that value is still
**1.25 V**, not 1.25 mmHg. If the pressure sensor is on `Dev1/ai1`, explicitly
extract that column and calibrate it before creating a separate
`timestamp,pressure` CSV for alignment. Setting `value_col=1` does not change
the reader's choice for this CSV.

### Example: read paired-line data

```python
from echopress.ingest import read_pstream

# Select the second numeric value in paired-line text records.
for record in read_pstream("measurements.pstream", value_col=1):
    print(record.timestamp, record.pressure)
```

## Units and calibration

`read_pstream` parses numeric values without applying calibration.
`align` and `prepare-align` use those parsed values as labels without
automatically invoking `apply_calibration`. Setting calibration coefficients
or a display-unit label does not convert these records by itself.

For raw-voltage measurements:

1. Identify the sensor's source column and preserve the original timestamps
   and acquisition log.
2. Apply the verified calibration for that channel. The affine helper
   `echopress.core.calibration.apply_calibration` computes
   `pressure = alpha * voltage + beta`. For output in mmHg, `alpha` must have
   units mmHg/V and `beta` must have units mmHg. Default identity coefficients
   do not establish a physical pressure calibration.
3. Write a separate `timestamp,pressure` CSV using a recognised P-stream name
   and make it the selected input to indexing/alignment. Avoid leaving raw and
   calibrated alternatives ambiguously indexed for the same session.
4. Record the source channel, coefficients, output units, and calibration
   provenance with the dataset. Do not recalibrate values already converted
   to pressure.

The `calibrate` command accepts numeric arrays; it is not a direct converter
for a mixed timestamp-and-channel CSV. Extract the intended numeric trace
first, then preserve or reattach timestamps when writing the calibrated CSV.

## Timestamp parsing

`read_pstream` relies on `parse_timestamp` to recognise these grammars:

```python
from echopress.ingest import parse_timestamp

parse_timestamp("2023-08-19T16:24:03Z")       # ISO 8601, explicit UTC
parse_timestamp("2023-08-19 16:24:03.5")      # Space-separated date/time, treated as UTC
parse_timestamp("16:24:03.5")                # Today's UTC date
parse_timestamp("1692456243.5")              # Seconds since Unix epoch, UTC
parse_timestamp("M08-D19-H16-M24-S03-U.128")  # Current UTC year
```

Unsupported timestamp tokens raise a parsing error. Explicit dates and time
zones are preferable for archived experiments: time-only and custom MDHMSU
forms derive a date or year from the clock at parsing time. Space-separated
timestamps are treated as UTC, so local-clock acquisition timestamps need an
explicit, correct timezone conversion before alignment.
