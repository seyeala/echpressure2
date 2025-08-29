# Dataset Indexer

The `DatasetIndexer` scans a dataset directory and records pressure (`P`-stream) and oscilloscope (`O`-stream) files. Session IDs are taken from the full filename stem, so `VoltPrsr001.csv` and `sessionA.csv` become the session IDs `VoltPrsr001` and `sessionA` respectively. The raw stem is preserved to avoid any ambiguity.

Lookups are case-insensitive. Internally, the indexer stores a lowercase map of session IDs so queries like `indexer.get_pstreams("VoltPrsr001")` and `indexer.get_pstreams("voltprsr001")` return the same results.

P-stream CSV files are matched using configurable patterns. Each pattern may be a plain prefix or a regular expression. Matching is case-insensitive and falls back to prefix/substring checks if a pattern is not a valid regular expression.

```python
from echopress.ingest import DatasetIndexer, Settings

# Build an index over /data using both prefix and regex patterns
settings = Settings()
settings.ingest.pstream_csv_patterns = ["voltprsr", r"anotherpstream\d+"]
indexer = DatasetIndexer("/data", settings=settings)

# Case-insensitive lookup
indexer.get_pstreams("VoltPrsr001") == indexer.get_pstreams("voltprsr001")
indexer.get_ostreams("sessionA") == indexer.get_ostreams("SESSIONA")

# Fallback behaviour: unknown session IDs return project-wide lists
indexer.get_pstreams("unknown")       # -> list of all P-stream paths
indexer.get_ostreams("unknown", fallback=False)  # -> []
```

`get_pstreams` and `get_ostreams` accept a `fallback` argument. When `fallback=True` (the default) and a session ID is missing, the indexer returns all files of that type in the project. Setting `fallback=False` yields an empty list instead.
