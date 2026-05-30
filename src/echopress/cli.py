from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

"""Command line interface for echopress using Typer."""

import json
import logging
import os
import random

import numpy as np
import click
import typer
from click.core import ParameterSource
from pydantic import ValidationError

from .adapters import get_adapter
from .core.calibration import apply_calibration
from .core.mapping import align_streams
from .core.macro_detector import MacroDetectorConfig, run_macro_detection
from .core.echo_peaks import EchoPeakConfig, run_echo_peak_detection
from .core.align_cleaner import AlignCleanerConfig, run_align_clean
from .core.peak_window_postprocess import PeakWindowPostprocessConfig, run_peak_window_postprocess
from .core.fft_export import FFTExportConfig, run_fft_postprocessed
from .core.config_io import apply_override, parse_override_value
from .core.qc_plots import QCPlotConfig, run_qc_plot
from .core.rmcpe import RMCPEConfig, run_rmcpe
from .core.tables import File2PressureMap, OscFiles, Signals, export_tables
from .core.tciml import TCIMLConfig, run_tciml
from .ml.dataset import PressureDatasetConfig, build_pressure_dataset
from .ml.train import PressureTrainConfig, run_train
from .ml.evaluate import PressureEvalConfig, run_evaluate
from .config import Settings, load_settings
from .ingest import DatasetIndexer, load_ostream, read_pstream
from ._typer import bad_parameter
from .pipeline.runner import PipelineError, resolve_active_align, run_prepare_align, summarize_pipeline_state, run_prepare_macro, run_prepare_echo, run_prepare_postprocess, run_prepare_fft, run_pipeline_full
from .pipeline.state import PipelineStateMigrationError

app = typer.Typer(help="Utilities for the echopress project")
logger = logging.getLogger(__name__)
SEGMENTATION_MODES = ("none", "rmcpe-tciml", "macro-windows")


def _parse_override_value(raw: str) -> object:
    try:
        return parse_override_value(raw)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    try:
        apply_override(data, keys, value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _apply_overrides(settings: Settings, overrides: List[str]) -> Settings:
    if not overrides:
        return settings
    data = settings.model_dump()
    for override in overrides:
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
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid configuration override: {exc}") from exc


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


def _resolve_align_table(
    settings: Settings, base_root: Path, override: Optional[Path] = None
) -> Path:
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

    settings = _apply_overrides(settings, set_overrides)

    if dataset_root is not None:
        settings = settings.model_copy(
            update={
                "dataset": settings.dataset.model_copy(
                    update={"root": str(dataset_root)}
                )
            }
        )

    if adapter_name is not None:
        settings = settings.model_copy(
            update={
                "adapter": settings.adapter.model_copy(update={"name": adapter_name})
            }
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
        "pstreams": {
            sid: [str(p) for p in paths] for sid, paths in indexer.pstreams.items()
        },
        "ostreams": {
            sid: [str(o) for o in paths] for sid, paths in indexer.ostreams.items()
        },
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
    data = (
        np.loadtxt(input, delimiter=",") if input.endswith(".csv") else np.load(input)
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

    all_pstreams = [
        p for paths in index_data.get("pstreams", {}).values() for p in paths
    ]

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


@app.command("revise-align")
def revise_align(
    align_table: Path = typer.Option(
        ...,
        "--align-table",
        dir_okay=False,
        file_okay=True,
        exists=True,
        help="Input alignment JSON table.",
    ),
    remove_list: Path = typer.Option(
        ...,
        "--remove-list",
        dir_okay=False,
        file_okay=True,
        exists=True,
        help="JSON/TXT/CSV list of datapoints to remove.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        dir_okay=False,
        file_okay=True,
        exists=False,
        help="Output revised alignment JSON.",
    ),
    match_key: str = typer.Option(
        "path",
        "--match-key",
        help=(
            "How to match remove-list entries to alignment rows. "
            "Options: path, path_basename, file_stamp, sid, sid_file_stamp, row_index."
        ),
    ),
    invert: bool = typer.Option(
        False,
        "--invert",
        help="Invert selection: keep only rows in remove-list instead of removing them.",
    ),
) -> None:
    """Revise an alignment table by removing rows listed in a remove-list."""

    from .core.alignment_edit import revise_alignment_by_remove_list

    summary = revise_alignment_by_remove_list(
        align_table=align_table,
        remove_list=remove_list,
        output=output,
        match_key=match_key,
        invert=invert,
    )

    typer.echo(json.dumps(summary, indent=2))


@app.command("flag-low-peak")
def flag_low_peak(
    dataset_root: Path = typer.Option(
        ...,
        "--dataset-root",
        dir_okay=True,
        file_okay=False,
        exists=True,
        help="Dataset root used to resolve file paths inside align.json.",
    ),
    align_table: Path = typer.Option(
        ...,
        "--align-table",
        dir_okay=False,
        file_okay=True,
        exists=True,
        help="Input alignment JSON table.",
    ),
    output_list: Path = typer.Option(
        ...,
        "--output-list",
        "-o",
        dir_okay=False,
        file_okay=True,
        exists=False,
        help="Output JSON remove-list.",
    ),
    channel: int = typer.Option(
        0,
        "--channel",
        help="Waveform channel index to inspect.",
    ),
    baseline_samples: Optional[int] = typer.Option(
        None,
        "--baseline-samples",
        help="Use first X samples as baseline window.",
    ),
    baseline_seconds: Optional[float] = typer.Option(
        None,
        "--baseline-seconds",
        help="Use first X seconds as baseline window; requires valid timestamps.",
    ),
    threshold_multiplier: float = typer.Option(
        1.0,
        "--threshold-multiplier",
        help=(
            "Remove if max(abs(signal)) <= multiplier * mean(abs(baseline_window)). "
            "Use 1.0 for literal requested rule; use 2-5 for stricter peak detection."
        ),
    ),
    include_missing: bool = typer.Option(
        True,
        "--include-missing/--skip-missing",
        help="Include missing files in the removal list.",
    ),
) -> None:
    """Create a remove-list for files whose peak amplitude is not above baseline."""

    if baseline_samples is None and baseline_seconds is None:
        raise typer.BadParameter(
            "Specify either --baseline-samples or --baseline-seconds"
        )

    if baseline_samples is not None and baseline_seconds is not None:
        raise typer.BadParameter(
            "Use only one of --baseline-samples or --baseline-seconds"
        )

    from .core.amplitude_filter import build_low_peak_remove_list

    summary = build_low_peak_remove_list(
        align_table=align_table,
        dataset_root=dataset_root,
        output_list=output_list,
        channel=channel,
        baseline_samples=baseline_samples,
        baseline_seconds=baseline_seconds,
        threshold_multiplier=threshold_multiplier,
        include_missing=include_missing,
    )

    typer.echo(json.dumps(summary, indent=2))


@app.command()
def detect_windows(
    macro_k_min: int = typer.Option(2, "--macro-k-min", help="Minimum macro window multiplier K."),
    macro_k_max: int = typer.Option(12, "--macro-k-max", help="Maximum macro window multiplier K."),
    max_env_points: int = typer.Option(55000, "--max-env-points", help="Envelope block cap for period detection."),
    peak_distance: int = typer.Option(1, "--peak-distance", help="Minimum peak spacing in envelope blocks."),
    macro_min_period: float = typer.Option(3.0, "--macro-min-period", help="Minimum macro period in samples."),
    first_peak_min_prominence: float = typer.Option(0.0, "--first-peak-min-prominence", help="Minimum prominence for first-peak transitions."),
    first_peak_max_residual: float = typer.Option(9999.0, "--first-peak-max-residual", help="Maximum allowed first-peak residual."),
    signature_peak_width: int = typer.Option(8, "--signature-peak-width", help="Peak half-width used by signature extraction."),
    signature_chunk_size: int = typer.Option(4096, "--signature-chunk-size", help="Chunk size for signature extraction."),
    signature_chunk_overlap: int = typer.Option(256, "--signature-chunk-overlap", help="Chunk overlap for signature extraction."),
    diagnostics_out: Optional[Path] = typer.Option(None, "--diagnostics-out", help="Write diagnostics JSON to this path."),
) -> None:
    """Legacy diagnostics-only window config dump. Use detect-macro-windows for actual detection."""

    diagnostics = {
        "macro_k_bounds": [macro_k_min, macro_k_max],
        "block_sizes": {"max_env_points": max_env_points, "peak_distance": peak_distance},
        "macro_min_period": macro_min_period,
        "first_peak_transition": {
            "min_prominence": first_peak_min_prominence,
            "max_residual": first_peak_max_residual,
        },
        "signature": {
            "peak_width": signature_peak_width,
            "chunk_size": signature_chunk_size,
            "chunk_overlap": signature_chunk_overlap,
        },
    }
    if diagnostics_out is not None:
        diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    typer.echo(json.dumps(diagnostics, indent=2))


@app.command("detect-macro-windows")
def detect_macro_windows(
    dataset_root: Path = typer.Option(..., "--dataset-root", dir_okay=True, file_okay=False, exists=True),
    align_table: Path = typer.Option(..., "--align-table", dir_okay=False, file_okay=True, exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", dir_okay=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    channel: int = typer.Option(0, "--channel"),
    k_min: int = typer.Option(1, "--k-min"),
    k_max: int = typer.Option(20, "--k-max"),
    force_k: Optional[int] = typer.Option(None, "--force-k"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    npz_only: bool = typer.Option(True, "--npz-only/--no-npz-only"),
    block_size: int = typer.Option(10000, "--block-size"),
    envelope_window: int = typer.Option(9, "--envelope-window"),
    pre_span: int = typer.Option(4, "--pre-span"),
    post_span: int = typer.Option(10, "--post-span"),
    raw_max_abs_min: float = typer.Option(100.0, "--raw-max-abs-min"),
    max_alignment_error_s: Optional[float] = typer.Option(1.0, "--max-alignment-error-s"),
    pr_min: Optional[float] = typer.Option(None, "--pr-min"),
    pr_max: Optional[float] = typer.Option(None, "--pr-max"),
    first_peak_search_frac: float = typer.Option(0.40, "--first-peak-search-frac"),
    backward_full_windows: bool = typer.Option(True, "--backward-full-windows/--no-backward-full-windows"),
    snap_tol_frac: float = typer.Option(0.12, "--snap-tol-frac"),
    write_signatures: bool = typer.Option(False, "--write-signatures/--no-write-signatures"),
    signature_left: int = typer.Option(1000, "--signature-left"),
    signature_right: int = typer.Option(12000, "--signature-right"),
    signature_chunk_size: int = typer.Option(4096, "--signature-chunk-size"),
    plot_diagnostics: bool = typer.Option(True, "--plot-diagnostics/--no-plot-diagnostics"),
    progress_every: int = typer.Option(25, "--progress-every"),
    quiet: bool = typer.Option(False, "--quiet/--no-quiet"),
) -> None:
    cfg = MacroDetectorConfig(
        dataset_root=dataset_root,
        align_table=align_table,
        output_dir=output_dir,
        config=config,
        channel=channel,
        k_min=k_min,
        k_max=k_max,
        force_k=force_k,
        max_files=max_files,
        npz_only=npz_only,
        block_size=block_size,
        envelope_window=envelope_window,
        pre_span=pre_span,
        post_span=post_span,
        raw_max_abs_min=raw_max_abs_min,
        max_alignment_error_s=max_alignment_error_s,
        pr_min=pr_min,
        pr_max=pr_max,
        first_peak_search_frac=first_peak_search_frac,
        backward_full_windows=backward_full_windows,
        snap_tol_frac=snap_tol_frac,
        write_signatures=write_signatures,
        signature_left=signature_left,
        signature_right=signature_right,
        signature_chunk_size=signature_chunk_size,
        plot_diagnostics=plot_diagnostics,
        progress_every=progress_every,
        quiet=quiet,
    )
    summary = run_macro_detection(cfg)
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("detect-echo-peaks")
def detect_echo_peaks(
    detection_dir: Path = typer.Option(..., "--detection-dir", dir_okay=True, file_okay=False, exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", dir_okay=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    channel: Optional[int] = typer.Option(None, "--channel"),
    use_registered: Optional[bool] = typer.Option(None, "--use-registered/--all-first-peaks"),
    zero_before_us: Optional[float] = typer.Option(None, "--zero-before-us"),
    zero_after_us: Optional[float] = typer.Option(None, "--zero-after-us"),
    zero_before_samples: Optional[int] = typer.Option(None, "--zero-before-samples"),
    zero_after_samples: Optional[int] = typer.Option(None, "--zero-after-samples"),
    hilbert_frac: Optional[float] = typer.Option(None, "--hilbert-frac"),
    min_prominence_rel: Optional[float] = typer.Option(None, "--min-prominence-rel"),
    min_height_rel: Optional[float] = typer.Option(None, "--min-height-rel"),
    min_distance_samples: Optional[int] = typer.Option(None, "--min-distance-samples"),
    refine_radius_samples: Optional[int] = typer.Option(None, "--refine-radius-samples"),
    max_peaks_per_window: Optional[int] = typer.Option(None, "--max-peaks-per-window"),
    save_cleaned_windows: Optional[bool] = typer.Option(None, "--save-cleaned-windows/--no-save-cleaned-windows"),
    progress_every: Optional[int] = typer.Option(None, "--progress-every"),
    quiet: Optional[bool] = typer.Option(None, "--quiet/--no-quiet"),
) -> None:
    """Detect secondary non-first echo peaks from stored macro-window first peaks using Hilbert/HTE."""
    cfg = EchoPeakConfig(
        detection_dir=detection_dir, output_dir=output_dir, config=config, channel=channel, use_registered=use_registered,
        zero_before_us=zero_before_us, zero_after_us=zero_after_us, zero_before_samples=zero_before_samples,
        zero_after_samples=zero_after_samples, hilbert_frac=hilbert_frac, min_prominence_rel=min_prominence_rel,
        min_height_rel=min_height_rel, min_distance_samples=min_distance_samples, refine_radius_samples=refine_radius_samples,
        max_peaks_per_window=max_peaks_per_window, save_cleaned_windows=save_cleaned_windows, progress_every=progress_every,
        quiet=quiet,
    )
    summary = run_echo_peak_detection(cfg)
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("clean-align")
def clean_align(
    align_table: Path = typer.Option(..., "--align-table", dir_okay=False, file_okay=True, exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", dir_okay=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    alignment_error_max: Optional[float] = typer.Option(1.0, "--alignment-error-max"),
    pressure_min: Optional[float] = typer.Option(None, "--pressure-min"),
    pressure_max: Optional[float] = typer.Option(None, "--pressure-max"),
) -> None:
    summary = run_align_clean(AlignCleanerConfig(align_table=align_table, output_dir=output_dir, config=config, alignment_error_max=alignment_error_max, pressure_min=pressure_min, pressure_max=pressure_max))
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("postprocess-peak-windows")
def postprocess_peak_windows(
    macro_dir: Path = typer.Option(..., "--macro-dir", dir_okay=True, file_okay=False, exists=True),
    echo_dir: Path = typer.Option(..., "--echo-dir", dir_okay=True, file_okay=False, exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", dir_okay=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    channel: int = typer.Option(0, "--channel"),
    zero_first_pulse_us: float = typer.Option(2.0, "--zero-first-pulse-us"),
    peak_neighbor_us: float = typer.Option(0.0, "--peak-neighbor-us"),
    gain_clip_min: float = typer.Option(0.25, "--gain-clip-min"),
    gain_clip_max: float = typer.Option(4.0, "--gain-clip-max"),
    use_last_common_windows: bool = typer.Option(True, "--use-last-common-windows/--use-first-common-windows"),
    window_mode: str = typer.Option("peak-to-peak", "--window-mode", help="Windowing mode: peak-to-peak | global-periodic-common"),
    window_anchor: str = typer.Option("first", "--window-anchor", help="Anchor for global-periodic-common: first | last"),
    window_output_layout: str = typer.Option("period-rows", "--window-output-layout", help="Global periodic output layout: period-rows | continuous-train"),
    use_registered_first_peaks: bool = typer.Option(False, "--use-registered-first-peaks/--no-use-registered-first-peaks"),
    require_registered_first_peaks: bool = typer.Option(False, "--require-registered-first-peaks/--no-require-registered-first-peaks"),
    periodicity_tolerance_frac: float = typer.Option(0.12, "--periodicity-tolerance-frac"),
    max_common_windows: Optional[int] = typer.Option(None, "--max-common-windows"),
    window_length_samples: Optional[int] = typer.Option(None, "--window-length-samples"),
    plan_only: bool = typer.Option(False, "--plan-only/--write-waveforms"),
    write_legacy_aliases: bool = typer.Option(False, "--write-legacy-aliases/--no-write-legacy-aliases"),
    strict_legacy_aliases: bool = typer.Option(False, "--strict-legacy-aliases/--no-strict-legacy-aliases"),
) -> None:
    if window_mode not in {"peak-to-peak", "global-periodic-common"}:
        raise typer.BadParameter("window_mode must be one of: peak-to-peak, global-periodic-common", param_hint="--window-mode")
    if window_anchor not in {"first", "last"}:
        raise typer.BadParameter("window_anchor must be one of: first, last", param_hint="--window-anchor")
    if window_output_layout not in {"period-rows", "continuous-train"}:
        raise typer.BadParameter("window_output_layout must be one of: period-rows, continuous-train", param_hint="--window-output-layout")

    summary = run_peak_window_postprocess(PeakWindowPostprocessConfig(
        macro_dir=macro_dir,
        echo_dir=echo_dir,
        output_dir=output_dir,
        config=config,
        channel=channel,
        zero_first_pulse_us=zero_first_pulse_us,
        peak_neighbor_us=peak_neighbor_us,
        gain_clip_min=gain_clip_min,
        gain_clip_max=gain_clip_max,
        use_last_common_windows=use_last_common_windows,
        window_mode=window_mode,
        window_anchor=window_anchor,
        window_output_layout=window_output_layout,
        use_registered_first_peaks=use_registered_first_peaks,
        require_registered_first_peaks=require_registered_first_peaks,
        periodicity_tolerance_frac=periodicity_tolerance_frac,
        max_common_windows=max_common_windows,
        window_length_samples=window_length_samples,
        plan_only=plan_only,
        write_legacy_aliases=write_legacy_aliases,
        strict_legacy_aliases=strict_legacy_aliases,
    ))
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("fft-postprocessed")
def fft_postprocessed(
    postprocess_dir: Path = typer.Option(..., "--postprocess-dir", dir_okay=True, file_okay=False, exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", dir_okay=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    source_product: Optional[str] = typer.Option(None, "--source-product"),
    fft_mode: str = typer.Option("full", "--fft-mode"),
    n_fft: Optional[int] = typer.Option(None, "--n-fft"),
    output_bins: Optional[int] = typer.Option(None, "--output-bins"),
    fft_bins: Optional[int] = typer.Option(None, "--fft-bins"),
) -> None:
    summary = run_fft_postprocessed(FFTExportConfig(postprocess_dir=postprocess_dir, output_dir=output_dir, config=config, source_product=source_product, fft_bins=fft_bins, fft_mode=fft_mode, n_fft=n_fft, output_bins=output_bins))
    typer.echo(json.dumps(summary, indent=2, default=float))


@app.command("plot-macro-qc")
def plot_macro_qc(input_dir: Path = typer.Option(..., "--input-dir", dir_okay=True, file_okay=False, exists=True), output_dir: Path = typer.Option(..., "--output-dir", dir_okay=True, file_okay=False)) -> None:
    typer.echo(json.dumps(run_qc_plot(QCPlotConfig(stage="macro", input_dir=input_dir, output_dir=output_dir)), indent=2))


@app.command("plot-echo-qc")
def plot_echo_qc(input_dir: Path = typer.Option(..., "--input-dir", dir_okay=True, file_okay=False, exists=True), output_dir: Path = typer.Option(..., "--output-dir", dir_okay=True, file_okay=False)) -> None:
    typer.echo(json.dumps(run_qc_plot(QCPlotConfig(stage="echo", input_dir=input_dir, output_dir=output_dir)), indent=2))


@app.command("plot-postprocess-qc")
def plot_postprocess_qc(input_dir: Path = typer.Option(..., "--input-dir", dir_okay=True, file_okay=False, exists=True), output_dir: Path = typer.Option(..., "--output-dir", dir_okay=True, file_okay=False)) -> None:
    typer.echo(json.dumps(run_qc_plot(QCPlotConfig(stage="postprocess", input_dir=input_dir, output_dir=output_dir)), indent=2))


@app.command("plot-fft-qc")
def plot_fft_qc(input_dir: Path = typer.Option(..., "--input-dir", dir_okay=True, file_okay=False, exists=True), output_dir: Path = typer.Option(..., "--output-dir", dir_okay=True, file_okay=False)) -> None:
    typer.echo(json.dumps(run_qc_plot(QCPlotConfig(stage="fft", input_dir=input_dir, output_dir=output_dir)), indent=2))


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
    plot_max_points: Optional[int] = typer.Option(
        None,
        "--plot-max-points",
        help="Downsample plots to at most this many points per series.",
    ),
    plot_show: bool = typer.Option(
        True,
        "--plot-show/--no-plot-show",
        help="Display plots interactively (disable for headless runs).",
    ),
    set_overrides: List[str] = typer.Option(
        [],
        "--set",
        help="Override configuration values using dotted paths, e.g. adapter.period_est.fs=1000000",
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
    segmentation: str = typer.Option(
        "none",
        "--segmentation",
        help="Segmentation mode: none | rmcpe-tciml | macro-windows.",
        case_sensitive=False,
        click_type=click.Choice(SEGMENTATION_MODES, case_sensitive=False),
    ),
    use_rmcpe_tciml: Optional[bool] = typer.Option(
        None,
        "--use-rmcpe-tciml/--no-use-rmcpe-tciml",
        help="(Deprecated) Use --segmentation=rmcpe-tciml or --segmentation=none.",
        hidden=True,
    ),
    window_period_samples: Optional[float] = typer.Option(
        None,
        "--window-period-samples",
        help="Manual period override in samples. Alias: ENVELOPE_PERIOD_SAMPLES.",
    ),
) -> Optional[List[np.ndarray]]:
    """Apply adapters to files sampled from the dataset.

    The command loads the alignment tables generated by the ``align`` step and
    selects files whose pressure value ``p*`` falls within ``[pr_min, pr_max]``.
    A random subset of ``n`` files is processed using the adapter selected by
    ``--adapter``.  When ``--plot`` is provided the raw signal and adapter
    outputs are visualised using helper functions from :mod:`viz.plot_adapter`.
    Use ``--plot-save`` or ``--no-plot-show`` in Colab/CLI sessions to avoid
    blocking on GUI backends. ``--plot-max-points`` can be used to downsample
    long series before plotting.
    When ``--output`` is supplied the resulting feature vectors are written to
    a NumPy file at the given path.  The processed NumPy arrays are returned to
    the caller when executed from Python and ``--output`` is omitted, otherwise
    a summary is printed.
    """

    settings = _get_settings(ctx)
    settings = _apply_overrides(settings, set_overrides)

    adapter_name = adapter or settings.adapter.name
    pr_min = settings.adapter.pr_min if pr_min is None else pr_min
    pr_max = settings.adapter.pr_max if pr_max is None else pr_max
    n = settings.adapter.n if n is None else n
    if ctx.get_parameter_source("plot") is ParameterSource.DEFAULT:
        plot = settings.adapter.plot
    plot_max_points = (
        settings.adapter.plot_max_points if plot_max_points is None else plot_max_points
    )

    adapter_obj = get_adapter(adapter_name)
    fs = settings.adapter.period_est.fs
    f0 = settings.adapter.period_est.f0
    if window_period_samples is None:
        raw = os.environ.get("WINDOW_PERIOD_SAMPLES") or os.environ.get(
            "ENVELOPE_PERIOD_SAMPLES"
        )
        if raw:
            window_period_samples = float(raw)

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
    tciml_by_file: dict[str, list[dict[str, int]]] = {}
    seg_mode = segmentation.lower()
    if use_rmcpe_tciml is not None:
        typer.secho(
            "Deprecation warning: --use-rmcpe-tciml is deprecated; use --segmentation.",
            err=True,
            fg=typer.colors.YELLOW,
        )
        seg_mode = "rmcpe-tciml" if use_rmcpe_tciml else "none"

    if seg_mode == "macro-windows":
        typer.secho(
            "macro-windows segmentation selected; falling back to no pre-segmentation in adapt.",
            err=True,
            fg=typer.colors.YELLOW,
        )
    elif seg_mode == "rmcpe-tciml":
        raw_arrays: list[np.ndarray] = []
        raw_names: list[str] = []
        for path_str, _ in items:
            o_path = Path(path_str)
            arr = np.asarray(load_ostream(o_path).channels)
            arr = arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 0 else np.ravel(arr)
            raw_arrays.append(np.ravel(arr))
            raw_names.append(str(o_path))
        if raw_arrays:
            r_cfg = RMCPEConfig(T_min=2.0, T_max=max(3.0, float(max(map(len, raw_arrays)))))
            r_summary, r_df = run_rmcpe(raw_arrays, r_cfg)
            t_hat = float(window_period_samples) if window_period_samples else float(r_summary["T_hat"])
            if np.isfinite(t_hat) and t_hat > 0:
                tc_cfg = TCIMLConfig(
                    T_hat=t_hat,
                    T_error_samples=max(1.0, float(r_summary.get("T_error_samples", 1.0))),
                    peak_width_samples=max(1, int(round(0.05 * t_hat))),
                )
                marker_df = run_tciml(raw_arrays, r_summary, r_df, tc_cfg)
                marker_df["file_id"] = marker_df["file_id"].map(lambda x: raw_names[int(str(x).split("_")[-1])] if str(x).startswith("file_") else str(x))
                for rec in marker_df[marker_df["accepted"]].to_dict("records"):
                    tciml_by_file.setdefault(str(rec["file_id"]), []).append(rec)
                cycle_len = t_hat
                f0 = fs / t_hat if fs and t_hat else f0
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
        file_markers = tciml_by_file.get(str(o_path), [])
        if file_markers:
            segs: list[np.ndarray] = []
            for marker in file_markers:
                lo = max(0, int(marker["window_start_idx"]))
                hi = min(data.size, int(marker["window_end_idx"]) + 1)
                if hi > lo:
                    segs.append(data[lo:hi])
            if segs:
                data = np.concatenate(segs)
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
                _plot_adapter(
                    data,
                    result_arr,
                    save=save_path,
                    show=plot_show,
                    max_points=plot_max_points,
                )
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


@app.command("build-pressure-regression-dataset")
def build_pressure_regression_dataset(
    fft_dir: Path = typer.Option(..., "--fft-dir", file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, dir_okay=True),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    set_overrides: List[str] = typer.Option([], "--set"),
) -> None:
    summary = build_pressure_dataset(
        PressureDatasetConfig(fft_dir=fft_dir, output_dir=output_dir, config=config, overrides=set_overrides)
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("train-pressure-regressor")
def train_pressure_regressor(
    dataset_dir: Path = typer.Option(..., "--dataset-dir", file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, dir_okay=True),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    set_overrides: List[str] = typer.Option([], "--set"),
) -> None:
    run_train(PressureTrainConfig(dataset_dir=dataset_dir, output_dir=output_dir, config=config, overrides=set_overrides))
    typer.echo(json.dumps({"status": "ok", "output_dir": str(output_dir)}, indent=2))


@app.command("eval-pressure-regressor")
def eval_pressure_regressor(
    dataset_dir: Path = typer.Option(..., "--dataset-dir", file_okay=False, dir_okay=True),
    model_dir: Path = typer.Option(..., "--model-dir", file_okay=False, dir_okay=True),
    split: str = typer.Option("test", "--split"),
) -> None:
    run_evaluate(PressureEvalConfig(dataset_dir=dataset_dir, model_dir=model_dir, split=split))
    typer.echo(json.dumps({"status": "ok", "model_dir": str(model_dir), "split": split}, indent=2))


@app.command("train-pressure-baseline")
def train_pressure_baseline(
    fft_dir: Path = typer.Option(..., "--fft-dir", file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, dir_okay=True),
    config: Optional[Path] = typer.Option(None, "--config", dir_okay=False, file_okay=True),
    set_overrides: List[str] = typer.Option([], "--set"),
) -> None:
    dataset_dir = output_dir / "pressure_regression_dataset"
    model_dir = output_dir / "pressure_regressor_tf"
    build_pressure_dataset(PressureDatasetConfig(fft_dir=fft_dir, output_dir=dataset_dir, config=config, overrides=set_overrides))
    run_train(PressureTrainConfig(dataset_dir=dataset_dir, output_dir=model_dir, config=config, overrides=set_overrides))
    run_evaluate(PressureEvalConfig(dataset_dir=dataset_dir, model_dir=model_dir, split="test"))
    typer.echo(json.dumps({"status": "ok", "dataset_dir": str(dataset_dir), "model_dir": str(model_dir)}, indent=2))


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



@app.command("prepare-align")
def prepare_align(
    dataset_root: Path = typer.Option(..., "--dataset-root", file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(..., "--out-dir", file_okay=False, dir_okay=True),
    channel: int = typer.Option(0, "--channel"),
    baseline_samples: int = typer.Option(10000, "--baseline-samples"),
    threshold_multiplier: float = typer.Option(50.0, "--threshold-multiplier"),
    alignment_error_max: float = typer.Option(1.0, "--alignment-error-max"),
    mode: str = typer.Option("auto", "--mode"),
    force: bool = typer.Option(False, "--force/--no-force"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    try:
        result = run_prepare_align(dataset_root=dataset_root, out_dir=out_dir, channel=channel, baseline_samples=baseline_samples, threshold_multiplier=threshold_multiplier, alignment_error_max=alignment_error_max, mode=mode, force=force)
        typer.echo(json.dumps(result, indent=2 if not as_json else None))
    except PipelineError as exc:
        typer.echo(json.dumps({"status": "blocked", "can_continue": False, "error_message": str(exc), "next_action": "Fix reported issue then rerun prepare-align --mode resume"}))
        raise typer.Exit(code=1)
    except PipelineStateMigrationError as exc:
        payload = {"status": "blocked", "can_continue": False, "failed_stage": "load_pipeline_state", "error_message": str(exc), "next_action": "Run pipeline repair-state or move old pipeline_state.json aside"}
        typer.echo(json.dumps(payload))
        raise typer.Exit(code=1)


@app.command("pipeline-bootstrap")
def pipeline_bootstrap(
    dataset_root: Path = typer.Option(..., "--dataset-root", file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(..., "--out-dir", file_okay=False, dir_okay=True),
    mode: str = typer.Option("auto", "--mode"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    result = run_prepare_align(dataset_root=dataset_root, out_dir=out_dir, channel=0, baseline_samples=10000, threshold_multiplier=50.0, alignment_error_max=1.0, mode=mode, force=(mode=="force"))
    typer.echo(json.dumps(result, indent=2 if not as_json else None))


@app.command("pipeline-status")
def pipeline_status(
    out_dir: Path = typer.Option(..., "--out-dir", file_okay=False, dir_okay=True),
    as_json: bool = typer.Option(False, "--json"),
    allow_incomplete: bool = typer.Option(False, "--allow-incomplete"),
) -> None:
    result = summarize_pipeline_state(out_dir)
    typer.echo(json.dumps(result, indent=2 if not as_json else None))
    if result.get('status') != 'ready' and not allow_incomplete:
        raise typer.Exit(code=1)


@app.command("pipeline-doctor")
def pipeline_doctor(
    dataset_root: Path = typer.Option(..., "--dataset-root", file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(..., "--out-dir", file_okay=False, dir_okay=True),
) -> None:
    issues=[]
    if not dataset_root.exists():
        issues.append({"issue":"dataset path missing","fix":"Set --dataset-root to existing dataset path"})
    if str(dataset_root).startswith('\\content'):
        issues.append({"issue":"Windows-style \\content path in Colab-like env","fix":"Use /content/..."})
    if os.name == 'nt' and str(dataset_root).startswith('/content'):
        issues.append({"issue":"/content path on Windows","fix":"Use local drive path like D:/..."})
    if not out_dir.exists():
        issues.append({"issue":"out_dir does not exist","fix":"Run prepare-align to create outputs"})
    status = summarize_pipeline_state(out_dir)
    if status.get('status') == 'missing':
        issues.append({"issue":"out_dir has no pipeline_state.json","fix":"Run prepare-align --mode repair-state or --mode auto"})
    active = resolve_active_align(out_dir, dataset_root)
    if active.get('status') != 'ok':
        issues.append({"issue":"no valid alignment artifact","fix":"Run prepare-align --mode resume"})
    typer.echo(json.dumps({"status": "ok" if not issues else "issues", "issues": issues}, indent=2))



@app.command("prepare-macro")
def prepare_macro(dataset_root: Path = typer.Option(..., "--dataset-root"), out_dir: Path = typer.Option(..., "--out-dir"), run_mode: str = typer.Option("smoke", "--run-mode"), smoke_max_files: Optional[int] = typer.Option(5, "--smoke-max-files"), mode: str = typer.Option("auto", "--mode"), as_json: bool = typer.Option(False, "--json")) -> None:
    r=run_prepare_macro(dataset_root=dataset_root,out_dir=out_dir,run_mode=run_mode,smoke_max_files=smoke_max_files,mode=mode)
    typer.echo(json.dumps(r, indent=2 if not as_json else None))

@app.command("prepare-echo")
def prepare_echo(dataset_root: Path = typer.Option(..., "--dataset-root"), out_dir: Path = typer.Option(..., "--out-dir"), mode: str = typer.Option("auto", "--mode"), as_json: bool = typer.Option(False, "--json")) -> None:
    r=run_prepare_echo(dataset_root=dataset_root,out_dir=out_dir,mode=mode)
    typer.echo(json.dumps(r, indent=2 if not as_json else None))

@app.command("prepare-postprocess")
def prepare_postprocess(dataset_root: Path = typer.Option(..., "--dataset-root"), out_dir: Path = typer.Option(..., "--out-dir"), mode: str = typer.Option("auto", "--mode"), as_json: bool = typer.Option(False, "--json")) -> None:
    r=run_prepare_postprocess(dataset_root=dataset_root,out_dir=out_dir,mode=mode)
    typer.echo(json.dumps(r, indent=2 if not as_json else None))

@app.command("prepare-fft")
def prepare_fft(dataset_root: Path = typer.Option(..., "--dataset-root"), out_dir: Path = typer.Option(..., "--out-dir"), mode: str = typer.Option("auto", "--mode"), fft_bins: int = typer.Option(1024, "--fft-bins"), as_json: bool = typer.Option(False, "--json")) -> None:
    r=run_prepare_fft(dataset_root=dataset_root,out_dir=out_dir,mode=mode,fft_bins=fft_bins)
    typer.echo(json.dumps(r, indent=2 if not as_json else None))

@app.command("run-pipeline")
def run_pipeline(
    dataset_root: Path = typer.Option(..., "--dataset-root", file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(..., "--out-dir", file_okay=False, dir_okay=True),
    stages: str = typer.Option("align,macro,echo,postprocess,fft", "--stages"),
    run_mode: str = typer.Option("smoke", "--run-mode"),
    smoke_max_files: Optional[int] = typer.Option(5, "--smoke-max-files"),
    mode: str = typer.Option("auto", "--mode"),
    fft_bins: int = typer.Option(1024, "--fft-bins"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    selected = [s.strip() for s in stages.split(',') if s.strip()]
    try:
        result = run_pipeline_full(dataset_root=dataset_root, out_dir=out_dir, stages=selected, run_mode=run_mode, smoke_max_files=smoke_max_files, mode=mode, fft_bins=fft_bins)
        typer.echo(json.dumps(result, indent=2 if not as_json else None))
    except PipelineError as exc:
        typer.echo(json.dumps({"status":"blocked","can_continue":False,"error_message":str(exc)}))
        raise typer.Exit(code=1)

def main() -> None:
    """Execute the Typer application."""

    app()


if __name__ == "__main__":
    main()
