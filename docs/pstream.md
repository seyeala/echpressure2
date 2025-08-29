# P-stream files

P-streams record pressure measurements alongside timestamps. Files are
conventionally named `voltprsr{ID}.csv`—for example, `voltprsr001.csv`—so the
`DatasetIndexer` can extract the identifier and catalog them. Filename
patterns used to identify CSVs live in
`conf/ingest/default.yaml` under `pstream_csv_patterns` and may be extended to
match additional naming schemes.

`read_pstream` parses three formats:

* **Paired lines** – a timestamp line followed by a line of values. The desired
  pressure column is selected with `value_col` (0‑based index, default `2`
  meaning the third number). Commas or whitespace may separate the values.
* **Simple line** – `<timestamp> <pressure>` or `<timestamp>,<pressure>` on a
  single line.
* **Headered CSV** – a CSV file with `timestamp,pressure` header. Other CSV
  files fall back to the paired/simple text rules above.

```python
from echopress.ingest import DatasetIndexer, read_pstream

# Locate files like 'voltprsr001.csv'
indexer = DatasetIndexer("/data")
pstream_file = indexer.first_pstream("001")

# Iterate over timestamp/pressure rows, pulling the third value in a paired
# values line via value_col
for record in read_pstream(pstream_file, value_col=2):
    print(record.timestamp, record.pressure)
```

## Timestamp parsing

`read_pstream` relies on `parse_timestamp` to recognise several timestamp
grammars:

```python
from echopress.ingest import parse_timestamp

parse_timestamp("2023-08-19T16:24:03Z")      # ISO 8601
parse_timestamp("16:24:03.5")                # HH:MM:SS(.frac), today’s date
parse_timestamp("1692456243.5")              # seconds since epoch
parse_timestamp("M08-D19-H16-M24-S03-U.128") # custom MDHMSU form
```

When no grammar matches, `read_pstream` falls back to interpreting the token as
seconds since the Unix epoch.
