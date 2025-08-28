# P-stream CSVs

P-streams record pressure measurements alongside timestamps. Files are
conventionally named `voltprsr{ID}.csv`—for example, `voltprsr001.csv`—so the
`DatasetIndexer` can extract the identifier and catalog them. The default
configuration searches for this prefix, though additional patterns may be
supplied under `ingest.pstream_csv_patterns`.

The simplest P-stream CSV contains two columns: `timestamp` and `pressure`.
`read_pstream` understands this format and yields `PStreamRecord` objects. The
function attempts to parse each `timestamp` using the configured grammar and
falls back to interpreting the value as seconds since the Unix epoch when no
explicit format matches.

```python
from echopress.ingest import DatasetIndexer, read_pstream

# Locate files like 'voltprsr001.csv'
indexer = DatasetIndexer("/data")
pstream_file = indexer.first_pstream("001")

# Iterate over timestamp,pressure rows
for record in read_pstream(pstream_file):
    print(record.timestamp, record.pressure)
```
