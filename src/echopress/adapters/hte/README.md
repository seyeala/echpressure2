# HTE Adapter

The Hilbert Transform Envelope (`hte`) adapter computes the amplitude
envelope of the analytic signal for each segmented cycle.

## Current implementation

`HteAdapter.layer1` uses `cycle_synchronous_map` for fixed-length cycle
segmentation. In `layer2`, `hilbert_envelope` in `adapters/base.py` uses an
FFT/IFFT construction of the analytic signal and returns its magnitude:

```text
analytic_signal = cycle + i * Hilbert(cycle)
envelope = abs(analytic_signal)
```

The implementation retains DC and, for even-length cycles, the Nyquist bin,
doubles positive-frequency bins, and suppresses negative-frequency bins
before the inverse FFT. The result is returned as
`{"envelope": envelope}`, with the same `(n_cycles, cycle_len)` shape as
the input cycle matrix. It does not rotate, align, pool, or summarize the
envelopes.

## Time shifts and feature interpretation

**The envelope is not inherently shift-invariant.** Shifting a waveform
also shifts its envelope. In the ideal continuous case,
`envelope(x(t - tau)) = envelope(x)(t - tau)`; this is shift
equivariance, not invariance. The FFT-based implementation has the
corresponding property for circular shifts within a fixed cycle, while
finite-window boundaries and resegmentation can introduce further changes.

Removing the carrier phase does not remove the timing of amplitude
variations. If shift-invariant features are required, align the waveforms or
envelopes to a consistent reference, or choose an appropriate aggregation
whose shift behavior has been validated. The current HTE adapter does not
perform those additional steps.

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- S. O. Sadjadi and J. H. L. Hansen, "Hilbert envelope based features for robust speaker identification under reverberant mismatched conditions," in *Proc. IEEE ICASSP*, 5448–5451 (2011).
