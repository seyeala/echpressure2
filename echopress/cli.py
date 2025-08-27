"""Command line interface for the *echopress* toolkit.

The CLI is built on top of `Typer <https://typer.tiangolo.com/>`_ and uses
`Hydra <https://hydra.cc/>`_ for hierarchical configuration management.  Each
command loads configuration from the ``conf/`` directory located at the project
root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from export.datasets import build_dataset

app = typer.Typer(add_completion=False, help="Echopress command line interface")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf"


def load_config(config_name: str = "config") -> DictConfig:
    """Load a Hydra configuration from ``conf/``.

    Parameters
    ----------
    config_name:
        Name of the configuration file to compose.  The default corresponds to
        ``conf/config.yaml``.
    """

    # hydra.initialize requires a path relative to the caller's file
    rel_config_path = os.path.relpath(CONFIG_PATH, Path(__file__).parent)
    with initialize(version_base=None, config_path=rel_config_path):
        cfg = compose(config_name=config_name)
    return cfg


@app.command()
def ingest(
    config_name: str = typer.Option("config", help="Name of Hydra config to use"),
    save_csv: bool = typer.Option(False, help="Persist the combined dataset as CSV"),
    save_npz: bool = typer.Option(False, help="Persist the dataset as compressed NPZ"),
) -> None:
    """Ingest raw data and build an in-memory dataset."""

    cfg = load_config(config_name)
    dataset_cfg = cfg.dataset

    save_csv_path = Path(dataset_cfg.csv) if save_csv and "csv" in dataset_cfg else None
    save_npz_path = Path(dataset_cfg.npz) if save_npz and "npz" in dataset_cfg else None

    X, y = build_dataset(
        dataset_cfg.path,
        target=dataset_cfg.target,
        feature_columns=getattr(cfg.mapping, "features", None),
        save_csv=save_csv_path,
        save_npz=save_npz_path,
    )

    typer.echo(f"Dataset loaded: X={X.shape}, y={y.shape}")


@app.command()
def calibrate(config_name: str = typer.Option("config", help="Name of Hydra config")) -> None:
    """Run calibration step using configuration."""

    cfg = load_config(config_name)
    typer.echo(f"Calibration method: {cfg.calibration.method}")


@app.command()
def adapter(config_name: str = typer.Option("config", help="Name of Hydra config")) -> None:
    """Execute model adapter according to configuration."""

    cfg = load_config(config_name)
    typer.echo(f"Adapter: {cfg.adapter.name}")


@app.command()
def visualize(config_name: str = typer.Option("config", help="Name of Hydra config")) -> None:
    """Generate visualisations based on configuration."""

    cfg = load_config(config_name)
    typer.echo(f"Visualisation type: {cfg.viz.type}")


def main() -> None:  # pragma: no cover - thin wrapper
    """Entry point used by ``python -m echopress.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
