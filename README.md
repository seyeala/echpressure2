# echpressure2

## Configuration

Runtime configuration is managed with [Hydra](https://hydra.cc).  The default
configuration in `conf/config.yaml` composes several YAML groups under
`conf/`, including:

* `dataset` – paths to example O- and P-streams
* `calibration` – per-channel calibration coefficients
* `adapter` – parameters for signal adapters
* `viz` – options for plotting

By default, the library operates on channel `3`. Override this with
`calibration.channel` or the `ECHOPRESS_CHANNEL` environment variable.

Commands in `echopress.cli` are wrapped by ``hydra.main`` so overrides can be
passed directly on the command line. For example, to adjust calibration
parameters at runtime:

```bash
python -m echopress.cli calibrate data.npy -o out.npy calibration.alpha=2.0
```

The `Settings` dataclass remains available for functions that expect it. In the
CLI, values from Hydra's ``calibration`` section are converted into ``Settings``
instances for compatibility. ``Settings.from_env`` and
``echopress.config.load_settings`` still allow configuration via environment
variables or explicit files when Hydra is not desired.
