from __future__ import annotations

"""Command line interface for echopress built with Typer and Hydra."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from hydra import initialize, compose

from .ingest import load_ostream, read_pstream
from .core.calibration import apply_calibration
from .adapters import get_adapter
from .config import Settings

app = typer.Typer(help="Utilities for the echopress project")


def _load_cfg(overrides: List[str] | None = None):
    """Load Hydra configuration composed from ``conf/``."""
    config_dir = Path(__file__).resolve().parent.parent.parent / "conf"
    with initialize(config_path=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


@app.command()
def ingest(overrides: List[str] = typer.Option(None, help="Hydra overrides")) -> None:
    """Load O- and P-streams as specified by the dataset config."""
    cfg = _load_cfg(overrides)
    ostream = load_ostream(cfg.dataset.ostream)
    pstream = list(read_pstream(cfg.dataset.pstream))
    typer.echo(
        f"O-stream samples: {len(ostream.timestamps)}, P-stream records: {len(pstream)}"
    )


@app.command()
def calibrate(
    input: str,
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    overrides: List[str] = typer.Option(None, help="Hydra overrides"),
) -> None:
    """Apply calibration coefficients to a numeric array."""
    cfg = _load_cfg(overrides)
    data = np.loadtxt(input, delimiter=",") if input.endswith(".csv") else np.load(input)
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
def adapter(
    signal: str,
    output: str,
    overrides: List[str] = typer.Option(None, help="Hydra overrides"),
) -> None:
    """Run an adapter on ``signal`` and save its first output array."""
    cfg = _load_cfg(overrides)
    data = np.load(signal)
    adapter_obj = get_adapter(cfg.adapter.name)
    cycles = adapter_obj.layer1(data, fs=cfg.adapter.fs, f0=cfg.adapter.f0)
    outputs = adapter_obj.layer2(cycles, fs=cfg.adapter.fs)
    first_key = next(iter(outputs))
    np.save(output, outputs[first_key])


@app.command()
def viz(signal: str, overrides: List[str] = typer.Option(None, help="Hydra overrides")) -> None:
    """Visualise ``signal`` using matplotlib if available."""
    cfg = _load_cfg(overrides)
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


if __name__ == "__main__":
    app()
