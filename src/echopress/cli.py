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
from .config import Settings
from .ingest import load_ostream, read_pstream


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
    settings = Settings(
        alpha=cfg.calibration.alpha,
        beta=cfg.calibration.beta,
        channel=cfg.calibration.channel,
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

