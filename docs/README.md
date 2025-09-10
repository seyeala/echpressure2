# Documentation

The `align` command accepts additional options to handle window-mode O-streams:

* `--window-mode` – treat files as timestamped capture windows with no channels.
* `--duration` – length of each window in seconds (forwarded to `load_ostream`).
* `--base-year` – year used when parsing timestamps embedded in filenames.

When window mode is active, files containing only timestamps are noted but still
recorded so their paths appear in the exported alignment table.
