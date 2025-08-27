"""Plot input signals and their adapter outputs."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .helpers import auto_label, load_array, plot_series, save_or_show
from .styles import apply_style


def plot_adapter(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    input_label: str | None = None,
    output_label: str | None = None,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Visualise ``inp`` alongside ``out``.

    Parameters
    ----------
    inp, out:
        Input and adapter output signals.  They are truncated to the same
        length before plotting.
    input_label, output_label:
        Optional axis labels for the two series.
    save, show:
        Behaviour flags forwarded to :func:`save_or_show`.
    """

    n = min(len(inp), len(out))
    inp = inp[:n]
    out = out[:n]

    diff = out - inp

    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    plot_series(ax1, inp, label=input_label)
    plot_series(ax1, out, label=output_label)
    ax1.set_ylabel("Amplitude")
    if input_label and output_label:
        ax1.set_title(f"{input_label} vs {output_label}")

    plot_series(ax2, diff, label="difference")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Delta")

    save_or_show(fig, save, show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise adapter outputs")
    parser.add_argument("--input", required=True, help="Path to the input signal")
    parser.add_argument("--output", required=True, help="Path to the adapter output signal")
    parser.add_argument("--input-label", help="Label for the input signal")
    parser.add_argument("--output-label", help="Label for the adapter output signal")
    parser.add_argument("--save", help="Path to save the figure")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    inp = load_array(args.input)
    out = load_array(args.output)
    inp_label = args.input_label or auto_label(args.input)
    out_label = args.output_label or auto_label(args.output)

    plot_adapter(
        inp,
        out,
        input_label=inp_label,
        output_label=out_label,
        save=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()
