from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


LEGACY_CANONICAL_PRODUCT = {
    "product_name": "canonical_processed",
    "kind": "processed",
    "path": "secondary_peak_processed_waveforms.npy",
    "manifest": "secondary_peak_processed_manifest.csv",
    "summary": "secondary_peak_processed_summary.json",
}


def _validate_product(postprocess_dir: Path, product: dict[str, Any]) -> dict[str, Any]:
    waveform_path = postprocess_dir / str(product["path"])
    manifest_path = postprocess_dir / str(product["manifest"])
    summary_path = postprocess_dir / str(product["summary"])

    missing = [str(p) for p in (waveform_path, manifest_path, summary_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required source product files:\n" + "\n".join(missing))

    waveforms = np.load(waveform_path)
    if waveforms.ndim != 2:
        raise ValueError(f"{waveform_path.name} must be 2D [n_rows, n_samples], got shape={waveforms.shape}")

    manifest = pd.read_csv(manifest_path)
    if len(manifest) != int(waveforms.shape[0]):
        raise ValueError(
            f"row count mismatch: {manifest_path.name} has {len(manifest)} rows but waveforms has {waveforms.shape[0]}"
        )

    return {
        "product_name": str(product["product_name"]),
        "waveform_path": waveform_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "shape": [int(waveforms.shape[0]), int(waveforms.shape[1])],
        "kind": product.get("kind"),
        "window_mode": product.get("window_mode"),
        "window_output_layout": product.get("window_output_layout"),
        "horizontal_normalized": product.get("horizontal_normalized"),
        "vertical_normalized": product.get("vertical_normalized"),
        "secondary_peak_suppressed": product.get("secondary_peak_suppressed"),
        "gain_normalized": product.get("gain_normalized"),
    }


def resolve_waveform_product(postprocess_dir: Path, source_product: Optional[str] = None) -> dict[str, Any]:
    registry_path = postprocess_dir / "waveform_products.json"
    if not registry_path.exists():
        chosen = source_product or "canonical_processed"
        if chosen != "canonical_processed":
            raise FileNotFoundError(
                f"waveform_products.json is missing in {postprocess_dir}. Only source_product='canonical_processed' is supported in legacy fallback."
            )
        return _validate_product(postprocess_dir, LEGACY_CANONICAL_PRODUCT)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    products = registry.get("products", {})
    chosen = source_product or registry.get("default_fft_product")
    if not chosen:
        raise ValueError("waveform_products.json must define default_fft_product when source_product is omitted")
    if chosen not in products:
        raise KeyError(f"Unknown source_product '{chosen}'. Available products: {sorted(products.keys())}")
    return _validate_product(postprocess_dir, products[chosen])
