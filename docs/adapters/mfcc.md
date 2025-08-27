# MFCC Adapter

The Mel-Frequency Cepstral Coefficients (MFCC) adapter computes log-energy in mel-spaced frequency bands followed by a discrete cosine transform, producing a compact, shift-invariant spectral envelope. It is part of the second-layer transform adapters [Plan].

## Algorithm
1. Filter the magnitude spectrum into $B$ mel bands and compute energies $E_k$.
2. Take logarithms $Y_k = \ln E_k$ and apply the discrete cosine transform
   $$c_m = \sum_{k=1}^{B} Y_k \cos\left[\frac{\pi m}{B}(k-0.5)\right],\quad m=1,\ldots,M.$$
3. Use the first $M$ coefficients $c_m$ as the feature vector $x_F$.

Because only spectral magnitudes are used, the features are invariant to global shifts [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. Acoust. Speech Signal Process.* 28(4), 357–366 (1980).
