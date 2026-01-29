from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

"""Command line interface for echopress using Typer."""

import json
import logging
import random

import numpy as np
import typer
from click.core import ParameterSource
from pydantic import ValidationError

from .adapters import get_adapter
from .core.calibration import apply_calibration
from .core.mapping import align_streams
from .core.tables import File2PressureMap, OscFiles, Signals, export_tables
from .config import Settings, load_settings
from .ingest import DatasetIndexer, load_ostream, read_pstream
from ._typer import bad_parameter

app = typer.Typer(help="Utilities for the echopress project")
logger = logging.getLogger(__name__)


def _parse_override_value(raw: str) -> object:
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise typer.BadParameter(f"invalid JSON override value: {raw}") from None
    return raw


def _ensure_path(settings: Settings, keys: List[str]) -> None:
    current: object = settings
    for key in keys[:-1]:
        if isinstance(current, dict):
            if key not in current:
                raise typer.BadParameter(f"unknown configuration key: {'.'.join(keys)}")
            current = current[key]
        else:
            if not hasattr(current, key):
                raise typer.BadParameter(f"unknown configuration key: {'.'.join(keys)}")
            current = getattr(current, key)
            if current is None:
                current = {}
    last = keys[-1]
    if isinstance(current, dict):
        if last not in current:
            raise typer.BadParameter(f"unknown configuration key: {'.'.join(keys)}")
    elif not hasattr(current, last):
        raise typer.BadParameter(f"unknown configuration key: {'.'.join(keys)}")


def _apply_override(data: Dict[str, object], keys: List[str], value: object) -> None:
    target = data
    for key in keys[:-1]:
        existing = target.get(key)
        if not isinstance(existing, dict):
            existing = {}
            target[key] = existing
        target = existing
    target[keys[-1]] = value


def _ensure_settings(obj: object) -> Settings:
    if isinstance(obj, Settings):
        return obj
    if isinstance(obj, dict):
        return Settings.model_validate(obj)
    raise typer.BadParameter("CLI context does not contain a Settings instance")


def _get_settings(ctx: typer.Context) -> Settings:
    if ctx.obj is None:
        ctx.obj = Settings()
    if not isinstance(ctx.obj, Settings):
        settings = _ensure_settings(ctx.obj)
        ctx.obj = settings
        return settings
    return ctx.obj


def _dataset_root(settings: Settings) -> Path:
    return Path(settings.dataset.root).expanduser()


def _resolve_align_table(settings: Settings, base_root: Path, override: Optional[Path] = None) -> Path:
    if override is not None:
        align_path = override
    else:
        align_value = settings.adapter.align_table
        if not align_value:
            align_path = base_root / "align.json"
        else:
            align_path = Path(align_value)
            if not align_path.is_absolute():
                align_path = base_root / align_path
    return align_path


@app.callback()
def init(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        dir_okay=False,
        file_okay=True,
        exists=False,
        help="Path to a YAML or JSON configuration file.",
    ),
    set_overrides: List[str] = typer.Option(
        [],
        "--set",
        help="Override configuration values using dotted paths, e.g. dataset.root=/data",
    ),
    dataset_root: Optional[Path] = typer.Option(
        None,
        "--dataset-root",
        dir_okay=True,
        file_okay=False,
        exists=False,
        help="Override the dataset.root value from the configuration file.",
    ),
    adapter_name: Optional[str] = typer.Option(
        None,
        "--adapter-name",
        help="Override the adapter.name value from the configuration file.",
    ),
    align_table: Optional[Path] = typer.Option(
        None,
        "--align-table",
        dir_okay=False,
        file_okay=True,
        exists=False,
        help="Override the adapter.align_table path.",
    ),
) -> None:
    """Initialise the Typer context with validated settings."""

    if config is not None and not config.exists():
        bad_parameter(f"configuration file not found: {config}", param_hint="--config")

    base_settings: Optional[Settings] = None
    if ctx.obj is not None:
        try:
            base_settings = _ensure_settings(ctx.obj)
        except typer.BadParameter:
            base_settings = None

    try:
        if config:
            settings = load_settings(config)
        elif base_settings is not None:
            settings = base_settings
        else:
            settings = Settings()
    except (FileNotFoundError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"failed to load configuration: {exc}") from exc

    if set_overrides:
        data = settings.model_dump()
        for override in set_overrides:
            if "=" not in override:
                bad_parameter(
                    "overrides must be of the form --set section.key=value",
                    param_hint="--set",
                )
            key, raw_value = override.split("=", 1)
            if not key:
                bad_parameter("override key cannot be empty", param_hint="--set")
            keys = key.split(".")
            _ensure_path(settings, keys)
            value = _parse_override_value(raw_value)
            _apply_override(data, keys, value)
        try:
            settings = Settings.model_validate(data)
        except ValidationError as exc:
            raise typer.BadParameter(f"invalid configuration override: {exc}") from exc

    if dataset_root is not None:
        settings = settings.model_copy(
            update={
                "dataset": settings.dataset.model_copy(update={"root": str(dataset_root)})
            }
        )

    if adapter_name is not None:
        settings = settings.model_copy(
            update={"adapter": settings.adapter.model_copy(update={"name": adapter_name})}
        )

    if align_table is not None:
        settings = settings.model_copy(
            update={
                "adapter": settings.adapter.model_copy(
                    update={"align_table": str(align_table)}
                )
            }
        )

    ctx.obj = settings


@app.command()
def index(
    ctx: typer.Context,
    cache: Optional[str] = typer.Option(None, "--cache", "-c"),
    dataset_root: Optional[Path] = typer.Option(
        None,
        "--dataset-root",
        dir_okay=True,
        file_okay=False,
        exists=False,
        help="Override the dataset.root configured in the settings.",
    ),
) -> None:
    """Scan the dataset directory and cache file paths.

    The resulting index is a JSON file containing mappings of session
    identifiers to available O-stream and P-stream files.  It can be reused
    by other commands to avoid repeatedly walking the dataset tree.
    """

    settings = _get_settings(ctx)
    root_path = Path(dataset_root) if dataset_root else _dataset_root(settings)

    if not root_path.exists():
        bad_parameter(
            f"dataset root not found: {root_path}", param_hint="--dataset-root"
        )

    indexer = DatasetIndexer(root_path, settings=settings)
    data = {
        "pstreams": {sid: [str(p) for p in paths] for sid, paths in indexer.pstreams.items()},
        "ostreams": {sid: [str(o) for o in paths] for sid, paths in indexer.ostreams.items()},
    }

    cache_path = Path(cache) if cache else root_path / "index.json"
    with open(cache_path, "w", encoding="utf8") as fh:
        json.dump(data, fh)
    typer.echo(
        f"Indexed dataset at {root_path}, cached {len(indexer.sessions())} sessions to {cache_path}"
    )


@app.command()
def ingest(ctx: typer.Context) -> None:
    """Load O- and P-streams as specified by the dataset config."""

    settings = _get_settings(ctx)
    dataset_cfg = settings.dataset
    if not hasattr(dataset_cfg, "ostream") or not hasattr(dataset_cfg, "pstream"):
        raise typer.BadParameter(
            "dataset configuration must define 'ostream' and 'pstream' entries"
        )
    ostream = load_ostream(dataset_cfg.ostream)
    pstream = list(read_pstream(dataset_cfg.pstream))
    typer.echo(
        f"O-stream samples: {len(ostream.timestamps)}, P-stream records: {len(pstream)}"
    )


@app.command()
def calibrate(
    ctx: typer.Context,
    input: str,
    output: Optional[str] = typer.Option(None, "--output", "-o"),
) -> None:
    """Apply calibration coefficients to a numeric array."""

    settings = _get_settings(ctx)
    data = np.loadtxt(input, delimiter=",") if input.endswith(".csv") else np.load(
        input
    )
    calibrated = apply_calibration(data, settings=settings)
    if output:
        if output.endswith(".csv"):
            np.savetxt(output, calibrated, delimiter=",")
        else:
            np.save(output, calibrated)
    else:
        typer.echo(" ".join(map(str, calibrated)))


@app.command()
def align(
    ctx: typer.Context,
    root: Optional[Path] = typer.Argument(
        None,
        help="Dataset root directory. Defaults to dataset.root from the settings.",
    ),
    export: Optional[Path] = typer.Option(None, "--export", "-e"),
    debug: bool = typer.Option(False, "--debug", help="Show tracebacks on failure"),
    window_mode: bool = typer.Option(
        False,
        "--window-mode/--no-window-mode",
        help="Treat O-streams as capture windows with no channel data",
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", help="Capture window duration in seconds"
    ),
    base_year: Optional[int] = typer.Option(
        None,
        "--base-year",
        help="Base year for timestamps embedded in window-mode filenames",
    ),
    dataset_root: Optional[Path] = typer.Option(
        None,
        "--dataset-root",
        dir_okay=True,
        file_okay=False,
        exists=False,
        help="Override the dataset root directory for this command.",
    ),
) -> None:
    """Align sessions listed in ``index.json`` under ``root``.

    The command expects an index digest produced by :func:`index`.  When
    ``index.json`` is missing it is built on the fly.  For each session the
    first O-stream/P-stream pair is aligned and the resulting tables are
    consolidated into ``align.json``.

    ``align`` supports optional "window mode" processing where O-stream files
    contain only timestamps.  Use ``--window-mode`` with ``--duration`` and
    ``--base-year`` to forward these parameters to :func:`load_ostream`.
    """

    settings = _get_settings(ctx)
    debug = debug

    if dataset_root is not None:
        base_root = Path(dataset_root)
    elif root is not None:
        base_root = Path(root)
    else:
        base_root = _dataset_root(settings)
    if not base_root.exists():
        bad_parameter(
            f"dataset root not found: {base_root}", param_hint="--dataset-root"
        )

    align_cfg = settings.align
    if ctx.get_parameter_source("window_mode") is ParameterSource.DEFAULT:
        window_mode = align_cfg.window_mode
    if duration is None:
        duration = align_cfg.duration
    if base_year is None:
        base_year = align_cfg.base_year

    index_path = base_root / "index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf8") as fh:
            index_data: Dict[str, Dict[str, List[str]]] = json.load(fh)
    else:
        indexer = DatasetIndexer(base_root, settings=settings)
        index_data = {
            "pstreams": {
                sid: [str(p) for p in indexer.get_pstreams(sid, fallback=False)]
                for sid in indexer.sessions()
            },
            "ostreams": {
                sid: [str(o) for o in indexer.get_ostreams(sid, fallback=False)]
                for sid in indexer.sessions()
            },
        }

    all_pstreams = [p for paths in index_data.get("pstreams", {}).values() for p in paths]

    signals = Signals()
    osc_files = OscFiles()
    fmap = File2PressureMap()

    for session, o_paths in sorted(index_data.get("ostreams", {}).items()):
        p_paths = index_data.get("pstreams", {}).get(session, []) or all_pstreams
        if not o_paths or not p_paths:
            continue
        o_path = Path(o_paths[0])
        if o_path.name in {"align.json", "index.json"}:
            continue
        p_path = Path(p_paths[0])

        try:
            ostream = load_ostream(
                o_path,
                window_mode=window_mode,
                duration_s=duration,
                base_year=base_year,
            )
            pstream = list(read_pstream(p_path))
            align_kwargs = {}
            if window_mode and settings.quality.reject_if_Ealign_gt_Omax:
                # Synthetic window captures do not contain signal samples, so the
                # midpoint derived from timestamps may fall outside the tight
                # tolerance configured for real data.  Disable strict rejection so
                # we still emit a pressure mapping for downstream commands.
                align_kwargs["reject_if_Ealign_gt_Omax"] = False
            result = align_streams(
                ostream,
                pstream,
                settings=settings,
                **align_kwargs,
            )
        except Exception as exc:
            msg = (
                f"Failed to align session {session} "
                f"(O-stream: {o_path}, P-stream: {p_path}): {exc}"
            )
            if debug:
                logger.exception(msg)
                raise
            typer.secho(msg, err=True)
            continue

        sid = ostream.session_id
        file_stamp = o_path.stem

        data = np.asarray(ostream.channels)
        if data.ndim == 2:
            if data.shape[1] > 0:
                data = data[:, 0]
            else:
                typer.echo(
                    f"O-stream {o_path} has zero channels; processing in window mode"
                )
                data = np.array([])
        data = np.asarray(data).reshape(-1)
        if data.size == 0:
            osc_files.add(sid, file_stamp, 0, str(o_path))
        else:
            for idx, value in enumerate(data):
                signals.add(sid, file_stamp, idx, float(value))
                osc_files.add(sid, file_stamp, idx, str(o_path))

        if result.mapping >= 0:
            pressure_value = pstream[result.mapping].pressure
            fmap.add(sid, file_stamp, pressure_value, alignment_error=result.E_align)

    tables = export_tables(signals, osc_files, fmap, tall=True)
    export_path = Path(export) if export else base_root / "align.json"
    with open(export_path, "w", encoding="utf8") as fh:
        json.dump(tables, fh, default=float)
    typer.echo(f"Exported tables to {export_path}")


@app.command()
def adapt(
    ctx: typer.Context,
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter-name",
        "--adapter",
        "-a",
        help="Adapter implementation to execute.",
    ),
    pr_min: Optional[float] = typer.Option(None, "--pr-min"),
    pr_max: Optional[float] = typer.Option(None, "--pr-max"),
    n: Optional[int] = typer.Option(None, "--n"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    plot: bool = typer.Option(False, "--plot/--no-plot"),
    plot_save: Optional[Path] = typer.Option(
        None,
        "--plot-save",
        help="Save adapter plots to this path (or directory) to avoid blocking.",
    ),
    plot_show: bool = typer.Option(
        True,
        "--plot-show/--no-plot-show",
        help="Display plots interactively (disable for headless runs).",
    ),
    dataset_root: Optional[Path] = typer.Option(
        None,
        "--dataset-root",
        dir_okay=True,
        file_okay=False,
        exists=False,
        help="Override the dataset root containing the alignment table.",
    ),
    align_table: Optional[Path] = typer.Option(
        None,
        "--align-table",
        dir_okay=False,
        file_okay=True,
        exists=False,
        help="Path to the alignment table. Defaults to <dataset root>/align.json.",
    ),
) -> Optional[List[np.ndarray]]:
    """Apply adapters to files sampled from the dataset.

    The command loads the alignment tables generated by the ``align`` step and
    selects files whose pressure value ``p*`` falls within ``[pr_min, pr_max]``.
    A random subset of ``n`` files is processed using the adapter selected by
    ``--adapter``.  When ``--plot`` is provided the raw signal and adapter
    outputs are visualised using helper functions from :mod:`viz.plot_adapter`.
    Use ``--plot-save`` or ``--no-plot-show`` in Colab/CLI sessions to avoid
    blocking on GUI backends.
    When ``--output`` is supplied the resulting feature vectors are written to
    a NumPy file at the given path.  The processed NumPy arrays are returned to
    the caller when executed from Python and ``--output`` is omitted, otherwise
    a summary is printed.
    """

    settings = _get_settings(ctx)

    adapter_name = adapter or settings.adapter.name
    pr_min = settings.adapter.pr_min if pr_min is None else pr_min
    pr_max = settings.adapter.pr_max if pr_max is None else pr_max
    n = settings.adapter.n if n is None else n
    if ctx.get_parameter_source("plot") is ParameterSource.DEFAULT:
        plot = settings.adapter.plot

    adapter_obj = get_adapter(adapter_name)
    fs = settings.adapter.period_est.fs
    f0 = settings.adapter.period_est.f0

    root_path = Path(dataset_root) if dataset_root else _dataset_root(settings)

    align_path = _resolve_align_table(settings, root_path, align_table)
    if not align_path.exists():
        bad_parameter(
            f"alignment table not found: {align_path}", param_hint="--align-table"
        )

    with open(align_path, "r", encoding="utf8") as fh:
        rows = json.load(fh)

    file_pressure: Dict[str, float] = {}
    for row in rows:
        path = row.get("path")
        pressure = row.get("pressure_value")
        if not path or pressure is None:
            continue
        if pr_min is not None and pressure < pr_min:
            continue
        if pr_max is not None and pressure > pr_max:
            continue
        # Deduplicate entries by file path
        file_pressure[str(path)] = float(pressure)

    if not file_pressure:
        typer.echo("No files matched the requested pressure range")
        return None

    seed = settings.adapter.seed
    rng = random.Random(seed)
    items = list(file_pressure.items())
    if n < len(items):
        items = rng.sample(items, n)

    outputs: List[np.ndarray] = []
    skipped = 0
    cycle_len = fs / f0 if f0 else None
    for path_str, pressure_value in items:
        o_path = Path(path_str)
        ostream = load_ostream(o_path)
        data = np.asarray(ostream.channels)
        if data.ndim == 2:
            data = data[:, 0]
        data = data.reshape(-1)
        if data.size == 0:
            typer.secho(
                f"Skipping {o_path.name}: file has no usable samples",
                err=True,
                fg=typer.colors.YELLOW,
            )
            skipped += 1
            continue
        if cycle_len is not None and data.size < cycle_len:
            typer.secho(
                (
                    f"Skipping {o_path.name}: {data.size} samples is shorter than one "
                    f"cycle ({cycle_len:.1f} samples)"
                ),
                err=True,
                fg=typer.colors.YELLOW,
            )
            skipped += 1
            continue
        cycles = adapter_obj.layer1(data, fs=fs, f0=f0)
        adapter_out = adapter_obj.layer2(cycles, fs=fs)
        first_key = next(iter(adapter_out))
        result_arr = adapter_out[first_key]
        if settings.adapter.output_length:
            result_arr = result_arr[..., : settings.adapter.output_length]
        outputs.append(result_arr)
        typer.echo(
            f"processed {o_path.name}: adapter={adapter_name} output_shape={result_arr.shape} pressure={pressure_value}"
        )
        if plot:
            try:  # pragma: no cover - optional dependency
                from viz.plot_adapter import plot_adapter as _plot_adapter

                save_path = None
                if plot_save:
                    if plot_save.exists() and plot_save.is_dir():
                        plot_dir = plot_save
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        save_path = plot_dir / f"{o_path.stem}_adapter.png"
                    elif plot_save.suffix == "":
                        plot_dir = plot_save
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        save_path = plot_dir / f"{o_path.stem}_adapter.png"
                    else:
                        save_path = plot_save
                _plot_adapter(data, result_arr, save=save_path, show=plot_show)
            except Exception:  # pragma: no cover - graceful fallback
                typer.echo("Plotting unavailable")

    if skipped:
        typer.secho(
            f"Skipped {skipped} file(s) due to insufficient samples.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    if output:
        if not outputs:
            typer.secho(
                "No feature arrays to save (all files skipped).",
                err=True,
                fg=typer.colors.YELLOW,
            )
            return None
        out_path = Path(output)
        np.save(out_path, np.stack(outputs))
        typer.echo(f"saved {len(outputs)} feature arrays to {out_path}")
        return None

    return outputs


@app.command()
def viz(ctx: typer.Context, signal: str) -> None:
    """Visualise ``signal`` using matplotlib if available."""

    settings = _get_settings(ctx)
    data = np.load(signal)
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(data)
        plt.title(settings.viz.title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        save_path = settings.viz.save
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception:  # pragma: no cover - graceful fallback
        typer.echo("matplotlib not available, printing summary statistics")
        typer.echo(f"mean={float(np.mean(data)):.3f} std={float(np.std(data)):.3f}")


def main() -> None:
    """Execute the Typer application."""

    app()


if __name__ == "__main__":
    main()
