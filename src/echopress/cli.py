from __future__ import annotations

"""Command line interface for echopress using Typer and Hydra."""

from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import typer
from omegaconf import DictConfig

from .adapters import get_adapter
from .core.calibration import apply_calibration
from .core.mapping import align_streams
from .core.tables import File2PressureMap, OscFiles, Signals, export_tables
from .config import Settings
from .ingest import DatasetIndexer, load_ostream, read_pstream
import json
import random


app = typer.Typer(help="Utilities for the echopress project")


@app.command()
def index(
    ctx: typer.Context,
    cache: Optional[str] = typer.Option(None, "--cache", "-c"),
) -> None:
    """Scan the dataset directory and cache file paths.

    The resulting index is a JSON file containing mappings of session
    identifiers to available O-stream and P-stream files.  It can be reused
    by other commands to avoid repeatedly walking the dataset tree.
    """

    cfg: DictConfig = ctx.obj
    root_cfg = getattr(cfg.dataset, "root", None)
    root_path: Path
    if isinstance(root_cfg, str):
        root_path = Path(root_cfg)
    else:
        # ``root`` may be a mapping with ``ostream``/``pstream`` entries; use
        # the O-stream root as the base directory since most datasets place
        # all files under a common tree.
        root_path = Path(getattr(root_cfg, "ostream", "."))

    indexer = DatasetIndexer(root_path)
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

    cfg: DictConfig = ctx.obj
    ostream = load_ostream(cfg.dataset.ostream)
    pstream = list(read_pstream(cfg.dataset.pstream))
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

    cfg: DictConfig = ctx.obj
    data = np.loadtxt(input, delimiter=",") if input.endswith(".csv") else np.load(
        input
    )
    settings = Settings()
    settings.calibration.alpha = list(cfg.calibration.alpha)
    settings.calibration.beta = list(cfg.calibration.beta)
    settings.pressure.scalar_channel = cfg.pressure.scalar_channel
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
    export: Optional[str] = typer.Option(None, "--export", "-e"),
) -> None:
    """Align O- and P-streams and populate mapping tables."""

    cfg: DictConfig = ctx.obj
    ostream = load_ostream(cfg.dataset.ostream)
    pstream = list(read_pstream(cfg.dataset.pstream))

    result = align_streams(
        ostream,
        pstream,
        tie_breaker=cfg.mapping.tie_breaker,
        O_max=cfg.mapping.O_max,
        W=cfg.mapping.W,
        kappa=cfg.mapping.kappa,
        reject_if_Ealign_gt_Omax=cfg.quality.reject_if_Ealign_gt_Omax,
    )

    sid = ostream.session_id
    file_path = Path(cfg.dataset.ostream)
    file_stamp = file_path.stem

    signals = Signals()
    osc_files = OscFiles()
    fmap = File2PressureMap()

    data = np.asarray(ostream.channels)
    if data.ndim == 2:
        data = data[:, 0]
    data = np.asarray(data).reshape(-1)
    for idx, value in enumerate(data):
        signals.add(sid, file_stamp, idx, float(value))
        osc_files.add(sid, file_stamp, idx, str(file_path))

    pressure_value = None
    if result.mapping >= 0:
        pressure_value = pstream[result.mapping].pressure
        fmap.add(sid, file_stamp, pressure_value, alignment_error=result.E_align)

    delta_p = result.diagnostics.get("delta_p")

    if export:
        tables = export_tables(signals, osc_files, fmap, tall=True)
        with open(export, "w", encoding="utf8") as fh:
            json.dump(tables, fh, default=float)
        typer.echo(f"Exported tables to {export}")
    else:
        typer.echo(
            f"Signals: {len(signals)}, OscFiles: {len(osc_files)}, File2PressureMap: {len(fmap)}"
        )
        if pressure_value is not None and delta_p is not None:
            typer.echo(
                f"Alignment index: {result.mapping}, E_align={result.E_align:.6f}, ΔP={delta_p:.6f}"
            )
        else:
            typer.echo(
                f"Alignment index: {result.mapping}, E_align={result.E_align:.6f}"
            )


@app.command()
def adapt(
    ctx: typer.Context,
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a"),
    pr_min: Optional[float] = typer.Option(None, "--pr-min"),
    pr_max: Optional[float] = typer.Option(None, "--pr-max"),
    n: Optional[int] = typer.Option(None, "--n"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    plot: Optional[bool] = typer.Option(None, "--plot/--no-plot"),
) -> Optional[List[np.ndarray]]:
    """Apply adapters to files sampled from the dataset.

    The command loads the alignment tables generated by the ``align`` step and
    selects files whose pressure value ``p*`` falls within ``[pr_min, pr_max]``.
    A random subset of ``n`` files is processed using the adapter selected by
    ``--adapter``.  When ``--plot`` is provided the raw signal and adapter
    outputs are visualised using helper functions from :mod:`viz.plot_adapter`.
    When ``--output`` is supplied the resulting feature vectors are written to
    a NumPy file at the given path.  The processed NumPy arrays are returned to
    the caller when executed from Python and ``--output`` is omitted, otherwise
    a summary is printed.
    """

    cfg: DictConfig = ctx.obj

    adapter_name = adapter or cfg.adapter.name
    pr_min = pr_min if pr_min is not None else getattr(cfg.adapter, "pr_min", None)
    pr_max = pr_max if pr_max is not None else getattr(cfg.adapter, "pr_max", None)
    n = n if n is not None else getattr(cfg.adapter, "n", 1)
    plot = plot if plot is not None else getattr(cfg.adapter, "plot", False)

    adapter_obj = get_adapter(adapter_name)
    fs = cfg.adapter.period_est.fs
    f0 = cfg.adapter.period_est.f0

    root_cfg = getattr(cfg.dataset, "root", None)
    if isinstance(root_cfg, str):
        root_path = Path(root_cfg)
    else:
        root_path = Path(getattr(root_cfg, "ostream", "."))

    # Location of alignment table exported by ``align`` command
    align_table = getattr(cfg.adapter, "align_table", None)
    if align_table is None:
        align_table = root_path / "align.json"
    align_table = Path(align_table)
    if not align_table.exists():
        raise typer.BadParameter(f"alignment table not found: {align_table}")

    with open(align_table, "r", encoding="utf8") as fh:
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

    seed = getattr(cfg.adapter, "seed", None)
    rng = random.Random(seed)
    items = list(file_pressure.items())
    if n < len(items):
        items = rng.sample(items, n)

    outputs: List[np.ndarray] = []
    for path_str, pressure_value in items:
        o_path = Path(path_str)
        ostream = load_ostream(o_path)
        data = np.asarray(ostream.channels)
        if data.ndim == 2:
            data = data[:, 0]
        data = data.reshape(-1)
        cycles = adapter_obj.layer1(data, fs=fs, f0=f0)
        adapter_out = adapter_obj.layer2(cycles, fs=fs)
        first_key = next(iter(adapter_out))
        result_arr = adapter_out[first_key]
        if cfg.adapter.output_length:
            result_arr = result_arr[..., : cfg.adapter.output_length]
        outputs.append(result_arr)
        typer.echo(
            f"processed {o_path.name}: adapter={adapter_name} output_shape={result_arr.shape} pressure={pressure_value}"
        )
        if plot:
            try:  # pragma: no cover - optional dependency
                from viz.plot_adapter import plot_adapter as _plot_adapter

                _plot_adapter(data, result_arr)
            except Exception:  # pragma: no cover - graceful fallback
                typer.echo("Plotting unavailable")

    if output:
        out_path = Path(output)
        np.save(out_path, np.stack(outputs))
        typer.echo(f"saved {len(outputs)} feature arrays to {out_path}")
        return None

    return outputs


@app.command()
def viz(ctx: typer.Context, signal: str) -> None:
    """Visualise ``signal`` using matplotlib if available."""

    cfg: DictConfig = ctx.obj
    data = np.load(signal)
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(data)
        plt.title(cfg.viz.get("title", "Signal"))
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        if cfg.viz.get("save"):
            plt.savefig(cfg.viz.save)
        else:
            plt.show()
    except Exception:  # pragma: no cover - graceful fallback
        typer.echo("matplotlib not available, printing summary statistics")
        typer.echo(f"mean={float(np.mean(data)):.3f} std={float(np.std(data)):.3f}")


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "conf"


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entrypoint executed by Hydra which dispatches to the Typer app."""

    app(obj=cfg)


if __name__ == "__main__":
    main()

