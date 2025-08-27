"""Plot reference and aligned signals as well as their difference."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .helpers import auto_label, load_array, plot_series, save_or_show
from .styles import apply_style


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise alignment between two signals")
    parser.add_argument("--reference", required=True, help="Path to the reference signal")
    parser.add_argument("--aligned", required=True, help="Path to the aligned signal")
    parser.add_argument("--ref-label", help="Label for the reference signal")
    parser.add_argument("--aligned-label", help="Label for the aligned signal")
    parser.add_argument("--save", help="Path to save the figure")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    ref = load_array(args.reference)
    ali = load_array(args.aligned)
    n = min(len(ref), len(ali))
    ref = ref[:n]
    ali = ali[:n]

    diff = ali - ref

    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ref_label = args.ref_label or auto_label(args.reference)
    ali_label = args.aligned_label or auto_label(args.aligned)

    plot_series(ax1, ref, label=ref_label)
    plot_series(ax1, ali, label=ali_label)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{ref_label} vs {ali_label}")

    plot_series(ax2, diff, label="difference")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Delta")

    save_or_show(fig, args.save, args.show)


if __name__ == "__main__":
    main()
