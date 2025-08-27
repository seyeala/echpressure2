from __future__ import annotations

"""Command line interface for echopress using Typer and Hydra."""

from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import typer
from omegaconf import DictConfig

from .adapters import get_adapter
from .core.calibration import apply_calibration
from .core.mapping import align_streams
from .core.tables import File2PressureMap, OscFiles, Signals, export_tables
from .config import Settings
from .ingest import load_ostream, read_pstream
import json


app = typer.Typer(help="Utilities for the echopress project")


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

    if result.mapping >= 0:
        pressure_value = pstream[result.mapping].pressure
        fmap.add(sid, file_stamp, pressure_value, alignment_error=result.E_align)

    if export:
        tables = export_tables(signals, osc_files, fmap, tall=True)
        with open(export, "w", encoding="utf8") as fh:
            json.dump(tables, fh, default=float)
        typer.echo(f"Exported tables to {export}")
    else:
        typer.echo(
            f"Signals: {len(signals)}, OscFiles: {len(osc_files)}, File2PressureMap: {len(fmap)}"
        )
        typer.echo(
            f"Alignment index: {result.mapping}, E_align={result.E_align:.6f}"
        )


@app.command()
def adapter(ctx: typer.Context, signal: str, output: str) -> None:
    """Run an adapter on ``signal`` and save its first output array."""

    cfg: DictConfig = ctx.obj
    data = np.load(signal)
    adapter_obj = get_adapter(cfg.adapter.name)
    fs = cfg.adapter.period_est.fs
    f0 = cfg.adapter.period_est.f0
    cycles = adapter_obj.layer1(data, fs=fs, f0=f0)
    outputs = adapter_obj.layer2(cycles, fs=fs)
    first_key = next(iter(outputs))
    result = outputs[first_key]
    if cfg.adapter.output_length:
        result = result[..., : cfg.adapter.output_length]
    np.save(output, result)


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

