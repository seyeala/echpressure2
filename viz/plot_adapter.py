"""Plot input signals and their adapter outputs."""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from .helpers import load_array, plot_series, save_or_show
from .styles import apply_style


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise adapter outputs")
    parser.add_argument("--input", required=True, help="Path to the input signal")
    parser.add_argument("--output", required=True, help="Path to the adapter output signal")
    parser.add_argument("--save", help="Path to save the figure")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    inp = load_array(args.input)
    out = load_array(args.output)
    n = min(len(inp), len(out))
    inp = inp[:n]
    out = out[:n]

    diff = out - inp

    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    plot_series(ax1, inp, label="input")
    plot_series(ax1, out, label="adapter output")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Input vs adapter output")

    plot_series(ax2, diff, label="difference")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Delta")

    save_or_show(fig, args.save, args.show)


if __name__ == "__main__":
    main()
