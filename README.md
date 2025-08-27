# echpressure2

## Configuration

Runtime behaviour is controlled through `echopress.config.Settings`. The dataclass
exposes fields for calibration (`alpha`, `beta`, `channel`), stream mapping
(`O_max`, `tie_breaker`) and derivative utilities (`W`, `kappa`).

Settings can be provided via environment variables prefixed with `ECHOPRESS_`
or loaded from a JSON or YAML file using `echopress.config.load_settings`.

Functions such as `apply_calibration` and `align_streams` accept a `Settings`
instance. The derivative estimators also consult these defaults when `W` or
`kappa` are not supplied explicitly.
