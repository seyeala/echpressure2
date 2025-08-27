"""Plot raw signal arrays from the command line."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .helpers import auto_label, load_array, plot_series, save_or_show
from .styles import apply_style


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one or more raw signals")
    parser.add_argument("signals", nargs="+", help="Paths to signal arrays (.npy or .csv)")
    parser.add_argument("--labels", nargs="*", help="Optional labels for each signal")
    parser.add_argument("--save", help="Path to save the figure")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.signals):
        parser.error("Number of labels must match number of signals")

    apply_style()
    fig, ax = plt.subplots()

    if args.labels:
        labels = args.labels
    else:
        labels = [auto_label(p) for p in args.signals]

    for path, label in zip(args.signals, labels):
        data = load_array(path)
        plot_series(ax, data, label=label)

    ax.set_title("Raw signals")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

    save_or_show(fig, args.save, args.show)


if __name__ == "__main__":
    main()
