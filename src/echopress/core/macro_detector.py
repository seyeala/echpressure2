from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd

from echopress.core.config_io import merge_config, write_resolved_config
from echopress.core.macro_windows import (
    FirstPeakConfig,
    MacroConfig,
    build_envelope,
    fit_macro_k_phase,
    flat_to_burst_score,
    generate_first_peak_candidates,
    select_periodic_first_peak_sequence,
)
from echopress.core.signatures import extract_peak_centered, write_signature_chunks
from echopress.core.window_consensus import aggregate_per_k, mad_outlier_flags, select_global_k
from echopress.ingest import load_ostream


@dataclass(frozen=True)
class MacroDetectorConfig:
    dataset_root: Path
    align_table: Path
    output_dir: Path
    config: Optional[Path] = None
    channel: int = 0
    k_min: int = 1
    k_max: int = 20
    force_k: Optional[int] = None
    max_files: Optional[int] = None
    npz_only: bool = True
    block_size: int = 10_000
    envelope_window: int = 9
    pre_span: int = 4
    post_span: int = 10
    phase_step: int = 1
    raw_max_abs_min: float = 100.0
    max_alignment_error_s: Optional[float] = 1.0
    pr_min: Optional[float] = None
    pr_max: Optional[float] = None
    first_peak_search_frac: float = 0.40
    first_peak_max_candidates: int = 5
    first_peak_periodicity_tolerance: float = 0.25
    backward_full_windows: bool = True
    snap_tol_frac: float = 0.12
    peak_spacing_outlier_mad: float = 3.5
    signature_left: int = 1000
    signature_right: int = 12000
    signature_chunk_size: int = 4096
    write_signatures: bool = False
    plot_diagnostics: bool = True
    progress_every: int = 25
    quiet: bool = False




def _eta_line(stage: str, i: int, total: int, start_time: float) -> str:
    elapsed = time.time() - start_time
    rate = i / elapsed if elapsed > 0 else 0.0
    remaining = total - i
    eta = remaining / rate if rate > 0 else float("nan")

    return (
        f"[{stage}] {i}/{total} files | "
        f"elapsed={elapsed/60:.1f} min | "
        f"rate={rate:.2f} files/s | "
        f"ETA={eta/60:.1f} min"
    )

def _robust_z(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = mad * 1.4826 if mad > 0 else float(np.std(x) + 1e-9)
    return (x - med) / max(scale, 1e-9)


def load_alignment_rows(cfg: MacroDetectorConfig) -> pd.DataFrame:
    rows = pd.read_json(cfg.align_table)

    if "path" not in rows.columns:
        raise ValueError(f"align table missing 'path'; columns={list(rows.columns)}")
    if "pressure_value" not in rows.columns:
        raise ValueError(f"align table missing 'pressure_value'; columns={list(rows.columns)}")

    rows = rows.dropna(subset=["path", "pressure_value"]).copy()

    if cfg.max_alignment_error_s is not None and "alignment_error" in rows.columns:
        rows["alignment_error"] = pd.to_numeric(rows["alignment_error"], errors="coerce")
        rows = rows[rows["alignment_error"].notna()]
        rows = rows[rows["alignment_error"] <= cfg.max_alignment_error_s]

    if cfg.pr_min is not None:
        rows = rows[rows["pressure_value"] >= cfg.pr_min]
    if cfg.pr_max is not None:
        rows = rows[rows["pressure_value"] <= cfg.pr_max]

    def resolve_path(p: object) -> str:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp.resolve())
        return str((cfg.dataset_root / pp).resolve())

    rows["path"] = rows["path"].map(resolve_path)

    if cfg.npz_only:
        rows = rows[rows["path"].astype(str).str.lower().str.endswith(".npz")]

    rows = rows[rows["path"].map(lambda p: Path(p).exists())]
    rows = rows.drop_duplicates(subset=["path"]).reset_index(drop=True)
    rows["file"] = rows["path"].map(lambda p: Path(p).name)

    if cfg.max_files is not None and cfg.max_files > 0 and len(rows) > cfg.max_files:
        rows = rows.sort_values("pressure_value").reset_index(drop=True)
        pick = np.linspace(0, len(rows) - 1, cfg.max_files).round().astype(int)
        rows = rows.iloc[pick].drop_duplicates(subset=["path"]).reset_index(drop=True)

    return rows


def coarse_macro_envelope(signal: np.ndarray, cfg: MacroDetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float).reshape(-1)
    n_blocks = max(1, int(np.ceil(x.size / cfg.block_size)))
    rms = np.zeros(n_blocks)
    max_abs = np.zeros(n_blocks)
    log_energy = np.zeros(n_blocks)
    centers = np.zeros(n_blocks, dtype=int)
    for i in range(n_blocks):
        lo, hi = i * cfg.block_size, min(x.size, (i + 1) * cfg.block_size)
        b = x[lo:hi]
        centers[i] = (lo + hi) // 2
        if b.size == 0:
            continue
        ab = np.abs(b)
        rms[i] = float(np.sqrt(np.mean(b * b)))
        max_abs[i] = float(np.max(ab))
        log_energy[i] = float(np.log(np.mean(b * b) + 1e-9))
    combined = 0.50 * _robust_z(log_energy) + 0.35 * _robust_z(rms) + 0.15 * _robust_z(max_abs)
    smooth = build_envelope(combined, mode="rms", window=cfg.envelope_window)
    return smooth, centers


def macro_transition_score(envelope: np.ndarray, cfg: MacroDetectorConfig) -> np.ndarray:
    score = np.array([flat_to_burst_score(envelope, i, pre_span=cfg.pre_span, post_span=cfg.post_span) for i in range(len(envelope))], dtype=float)
    smin, smax = float(np.min(score)), float(np.max(score))
    return np.zeros_like(score) if smax <= smin else (score - smin) / (smax - smin)


def score_file_k(envelope: np.ndarray, n_samples: int, cfg: MacroDetectorConfig) -> pd.DataFrame:
    ks = [cfg.force_k] if cfg.force_k else list(range(cfg.k_min, cfg.k_max + 1))
    rows = []
    for K in ks:
        step_bins = len(envelope) / float(K)
        m_cfg = MacroConfig(k_candidates=(step_bins,), phase_step=max(1, int(step_bins // 100)), envelope_mode="rms", envelope_window=cfg.envelope_window, pre_span=cfg.pre_span, post_span=cfg.post_span)
        fit = fit_macro_k_phase(envelope, m_cfg)
        rows.append({"K": int(K), "step_bins": float(step_bins), "phase_bins": int(fit.phase), "T_macro_guess_samples": float(n_samples / float(K)), "score": float(fit.score)})
    return pd.DataFrame(rows)


def select_dataset_global_k(all_k_scores: pd.DataFrame, cfg: MacroDetectorConfig) -> int:
    scores_by_k = {int(k): grp["score"].tolist() for k, grp in all_k_scores.groupby("K")}
    agg = aggregate_per_k(scores_by_k)
    return int(round(select_global_k(agg)))




def _resolve_config(cfg: MacroDetectorConfig) -> dict[str, object]:
    cli_values = {
        key: value
        for key, value in asdict(cfg).items()
        if key not in {"dataset_root", "align_table", "output_dir", "config"}
    }
    default_yml = Path(__file__).resolve().parents[3] / "configs" / "macro_windows.default.yml"
    resolved = merge_config(
        default_yaml_path=default_yml,
        user_yaml_path=Path(cfg.config) if cfg.config is not None else None,
        cli_values=cli_values,
    )
    resolved["dataset_root"] = str(Path(cfg.dataset_root))
    resolved["align_table"] = str(Path(cfg.align_table))
    resolved["output_dir"] = str(Path(cfg.output_dir))
    return resolved

def run_macro_detection(cfg: MacroDetectorConfig) -> dict:
    rcfg = _resolve_config(cfg)
    cfg = MacroDetectorConfig(
        dataset_root=Path(rcfg["dataset_root"]),
        align_table=Path(rcfg["align_table"]),
        output_dir=Path(rcfg["output_dir"]),
        **{k: v for k, v in rcfg.items() if k not in {"dataset_root", "align_table", "output_dir"}},
    )
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(rcfg, out / "detect-macro-windows_config.resolved.yml")
    diag_dir = out / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    align = load_alignment_rows(cfg)
    align.to_csv(out / "alignment_filtered.csv", index=False)

    if not cfg.quiet:
        ae = align["alignment_error"] if "alignment_error" in align.columns else None
        print(
            f"[detect-macro-windows] loaded {len(align)} files | "
            f"k={cfg.force_k if cfg.force_k is not None else f'{cfg.k_min}-{cfg.k_max}'} | "
            f"max_files={cfg.max_files} | "
            f"write_signatures={cfg.write_signatures}",
            flush=True,
        )
        if ae is not None and len(ae):
            print(
                f"[detect-macro-windows] alignment_error "
                f"min/median/max={float(ae.min()):.6g}/"
                f"{float(ae.median()):.6g}/"
                f"{float(ae.max()):.6g}",
                flush=True,
            )

    k_rows = []
    cache = []
    t0 = time.time()
    total = len(align)
    for i, row in align.reset_index(drop=True).iterrows():
        file_no = i + 1
        if not cfg.quiet and (
            file_no == 1
            or file_no % cfg.progress_every == 0
            or file_no == total
        ):
            print(_eta_line("macro-K pass", file_no, total, t0), flush=True)
        sig = np.asarray(load_ostream(Path(row.path)).channels)
        sig = sig[:, cfg.channel] if sig.ndim == 2 else sig.reshape(-1)
        if sig.size == 0 or np.max(np.abs(sig)) < cfg.raw_max_abs_min:
            continue
        env, centers = coarse_macro_envelope(sig, cfg)
        trans = macro_transition_score(env, cfg)
        kdf = score_file_k(env, len(sig), cfg)
        kdf["path"] = row.path
        k_rows.append(kdf)
        cache.append({
            "file_index": i,
            "path": row.path,
            "file": row.file,
            "pressure_value": row.pressure_value,
            "n_samples": len(sig),
            "env": env,
            "centers": centers,
            "trans": trans,
        })

    if not k_rows:
        raise RuntimeError(
            "No usable files after alignment filtering and raw_max_abs_min filtering. "
            "Check align table, channel, npz_only, and raw_max_abs_min."
        )

    all_k = pd.concat(k_rows, ignore_index=True)
    all_k.to_csv(out / "k_scores.csv", index=False)
    global_k = select_dataset_global_k(all_k, cfg)
    (out / "global_k_summary.json").write_text(json.dumps({"global_k": global_k, "n_files": int(len(cache))}, indent=2))

    windows_all = []
    peaks_all = []
    t1 = time.time()
    total2 = len(cache)
    for n, item in enumerate(cache, start=1):
        if not cfg.quiet and (
            n == 1
            or n % cfg.progress_every == 0
            or n == total2
        ):
            print(_eta_line("window/peak pass", n, total2, t1), flush=True)
        sig = np.asarray(load_ostream(Path(item["path"])).channels)
        sig = sig[:, cfg.channel] if sig.ndim == 2 else sig.reshape(-1)
        env = item["env"]
        trans = item["trans"]
        step_bins = len(env) / float(global_k)
        fit = fit_macro_k_phase(env, MacroConfig(k_candidates=(step_bins,), phase_step=max(1, int(step_bins // 100)), pre_span=cfg.pre_span, post_span=cfg.post_span))
        phase_samples = fit.phase * cfg.block_size
        T_guess = len(sig) / float(global_k)
        for j in range(global_k):
            expected = int(round(phase_samples + j * T_guess))
            radius = int(max(1, round(cfg.snap_tol_frac * T_guess / cfg.block_size)))
            c = int(np.clip(expected // cfg.block_size, 0, len(trans) - 1))
            lo, hi = max(0, c - radius), min(len(trans), c + radius + 1)
            b = lo + int(np.argmax(trans[lo:hi]))
            onset = int(min(len(sig) - 1, b * cfg.block_size))
            end = int(min(len(sig), onset + T_guess))
            windows_all.append({"path": item["path"], "file": item["file"], "file_index": item["file_index"], "pressure_value": item["pressure_value"], "macro_window_index": j, "macro_onset_idx": onset, "macro_window_start_idx": onset, "macro_window_end_idx_exclusive": end, "macro_transition_score": float(trans[b]), "n_samples": len(sig)})

    windows_df = pd.DataFrame(windows_all)
    windows_df.to_csv(out / "macro_window_table.csv", index=False)
    windows_df.groupby("path").size().reset_index(name="n_windows").to_csv(out / "macro_window_per_file.csv", index=False)

    for path, grp in windows_df.groupby("path"):
        sig = np.asarray(load_ostream(Path(path)).channels)
        sig = sig[:, cfg.channel] if sig.ndim == 2 else sig.reshape(-1)
        cands_per = []
        local = []
        for _, w in grp.sort_values("macro_window_index").iterrows():
            ws, we = int(w.macro_window_start_idx), int(w.macro_window_end_idx_exclusive)
            wl = max(1, we - ws)
            search_end = min(we, ws + int(round(cfg.first_peak_search_frac * wl)))
            seg = sig[ws:search_end]
            fp_cfg = FirstPeakConfig(k=float(wl), periodicity_tolerance=cfg.first_peak_periodicity_tolerance)
            cands = generate_first_peak_candidates(seg, fp_cfg)
            picks = [ws + int(c) for c in cands[: cfg.first_peak_max_candidates]]
            cands_per.append(tuple(picks))
            local.append((w, picks))
        seq = select_periodic_first_peak_sequence(cands_per, expected_k=float(np.median(np.diff(grp["macro_window_start_idx"])) if len(grp) > 1 else 1.0), tolerance=cfg.first_peak_periodicity_tolerance)
        chosen = list(seq.indices) if seq.indices else [p[1][0] if p[1] else int(p[0].macro_window_start_idx) for p in local]
        for idx, (w, picks) in enumerate(local):
            peak = int(chosen[idx] if idx < len(chosen) else (picks[0] if picks else w.macro_window_start_idx))
            val = float(sig[peak]) if 0 <= peak < len(sig) else 0.0
            peaks_all.append({"path": path, "file": w.file, "pressure_value": w.pressure_value, "file_index": int(w.file_index), "macro_window_index": int(w.macro_window_index), "macro_window_start_idx": int(w.macro_window_start_idx), "macro_window_end_idx_exclusive": int(w.macro_window_end_idx_exclusive), "first_peak_idx": peak, "first_peak_value": val, "first_peak_abs_value": abs(val), "offset_from_window_start": int(peak - w.macro_window_start_idx), "flat_left_score": float(flat_to_burst_score(np.abs(sig), peak, pre_span=max(1, cfg.block_size // 8), post_span=max(2, cfg.block_size // 4))), "transition_peak_score": float(w.macro_transition_score), "periodicity_error": float(seq.periodicity_error)})

    first_df = pd.DataFrame(peaks_all)
    first_df.to_csv(out / "first_peak_index.csv", index=False)

    spacings = []
    for _, grp in first_df.groupby("path"):
        p = np.sort(grp["first_peak_idx"].to_numpy())
        if len(p) > 1:
            spacings.extend(np.diff(p).tolist())
    spacings = np.asarray(spacings, dtype=float)
    flags = mad_outlier_flags(spacings, z_thresh=cfg.peak_spacing_outlier_mad)
    used = spacings[~flags] if spacings.size else spacings
    T_global = float(np.median(used)) if used.size else float("nan")
    pd.DataFrame({"spacing": spacings, "is_outlier": flags if spacings.size else []}).to_csv(out / "peak_spacing_distribution.csv", index=False)

    reg_rows = []
    counts = []
    for path, grp in first_df.groupby("path"):
        g = grp.sort_values("first_peak_idx").copy()
        n_samples = int(grp["macro_window_end_idx_exclusive"].max())
        valid = g[g["first_peak_idx"] + T_global <= n_samples]
        if valid.empty:
            continue
        anchor = float(valid["first_peak_idx"].max())
        selected = []
        m = 0
        while True:
            target = anchor - m * T_global
            if target < 0:
                break
            nearest_i = int(np.argmin(np.abs(g["first_peak_idx"].to_numpy() - target)))
            nearest = g.iloc[nearest_i]
            if abs(float(nearest.first_peak_idx) - target) <= cfg.snap_tol_frac * T_global:
                selected.append(int(nearest.first_peak_idx))
            m += 1
        selected = sorted(set(selected))
        counts.append({"path": path, "file": g.iloc[0].file, "full_window_count": len(selected)})
        g["used_for_backward_common_window"] = g["first_peak_idx"].isin(selected)
        reg_rows.append(g)

    reg_df = pd.concat(reg_rows, ignore_index=True) if reg_rows else first_df.assign(used_for_backward_common_window=False)
    counts_df = pd.DataFrame(counts)
    common = int(counts_df["full_window_count"].min()) if not counts_df.empty else 0
    final = []
    for path, grp in reg_df.groupby("path"):
        used_grp = grp[grp["used_for_backward_common_window"]].sort_values("first_peak_idx")
        if len(used_grp) >= common and common > 0:
            take = used_grp.iloc[-common:]
            keep = grp[~grp["used_for_backward_common_window"]]
            grp = pd.concat([keep, take], ignore_index=True)
        final.append(grp)
    reg_df = pd.concat(final, ignore_index=True) if final else reg_df
    reg_df.to_csv(out / "first_peak_index.registered.csv", index=False)
    counts_df.to_csv(out / "backward_full_window_counts.csv", index=False)

    if cfg.write_signatures:
        sig_rows = reg_df[reg_df.get("used_for_backward_common_window", False)].copy()
        blobs = []
        meta = []
        total_sig = len(sig_rows)
        t2 = time.time()
        for i, r in sig_rows.reset_index(drop=True).iterrows():
            if not cfg.quiet and (
                i + 1 == 1
                or (i + 1) % cfg.progress_every == 0
                or i + 1 == total_sig
            ):
                print(_eta_line("signature pass", i + 1, total_sig, t2), flush=True)
            sig = np.asarray(load_ostream(Path(r.path)).channels)
            sig = sig[:, cfg.channel] if sig.ndim == 2 else sig.reshape(-1)
            blobs.append(extract_peak_centered(sig, int(r.first_peak_idx), left=cfg.signature_left, right=cfg.signature_right))
            meta.append({"path": r.path, "first_peak_idx": int(r.first_peak_idx), "registered_window_index": i})
        if blobs:
            sig_arr = np.vstack(blobs)
            sig_dir = out / "signatures"
            idx_path = write_signature_chunks(sig_arr, sig_dir, chunk_size=cfg.signature_chunk_size)
            np.save(out / "signature_index.npy", np.load(idx_path, allow_pickle=True), allow_pickle=True)
            pd.DataFrame(meta).to_csv(out / "signature_index.csv", index=False)

    summary = {"T_global_samples": T_global, "common_full_windows": common, "global_k": global_k, "n_files_used": int(len(cache)), "method": "macro_flat_to_burst_then_backward_from_last_full_first_peak"}
    (out / "global_window_size.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
