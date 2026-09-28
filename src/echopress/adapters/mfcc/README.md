# MFCC Adapter

**Status: simplified cepstral approximation.** The existing `mfcc` adapter
identifier is retained, but the current implementation is not a complete
mel-frequency cepstral coefficient pipeline.

## Current implementation

The adapter first uses `cycle_synchronous_map` to form fixed-length cycles.
The `mfcc` helper in `adapters/base.py` then processes each cycle as follows:

1. Compute the one-sided FFT magnitude: `mag = abs(rfft(cycle))`.
2. Take the natural logarithm: `log_mag = log(mag + 1e-12)`.
3. Project the log magnitudes onto a type-II DCT-style cosine matrix over
   the linearly spaced FFT bins, using the unnormalized matrix defined in
   `_dct_matrix`.

The helper returns up to `n_mfcc` coefficients (default `13`), limited by the
number of FFT bins, including the zeroth coefficient. The adapter returns
these arrays under the `"mfcc"` key. For an input cycle matrix of shape
`(n_cycles, cycle_len)`, the output is
`(n_cycles, min(13, cycle_len // 2 + 1))` with the adapter's default call.

## Differences from full MFCC processing

There is no mel-spaced filter bank or mel-band energy calculation. The
implementation operates directly on log FFT magnitudes. Consequently, its
coefficients should be described as simplified cepstral features; they should
not be treated as numerically equivalent to standard MFCC outputs.

Mel-band filtering and a fully specified MFCC convention would require
additional implementation and validation. The references below provide
background for that method, not evidence that it is implemented here.

Magnitude spectra are invariant to circular shifts within a fixed cycle.
Arbitrary time shifts can change segmentation and boundary content, so the
complete adapter has no general time-shift-invariance guarantee.

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. Acoust. Speech Signal Process.* 28(4), 357–366 (1980).
