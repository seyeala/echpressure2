"""Plot reference and aligned signals as well as their difference."""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from .helpers import load_array, plot_series, save_or_show
from .styles import apply_style


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise alignment between two signals")
    parser.add_argument("--reference", required=True, help="Path to the reference signal")
    parser.add_argument("--aligned", required=True, help="Path to the aligned signal")
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

    plot_series(ax1, ref, label="reference")
    plot_series(ax1, ali, label="aligned")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Reference vs aligned")

    plot_series(ax2, diff, label="difference")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Delta")

    save_or_show(fig, args.save, args.show)


if __name__ == "__main__":
    main()
