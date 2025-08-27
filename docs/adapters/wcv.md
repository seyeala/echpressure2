# WCV Adapter

The Wavelet Coefficient Vector (WCV) adapter summarizes energy across wavelet scales and is part of the second-layer transform adapters [Plan].

## Algorithm
1. Apply a discrete wavelet transform to obtain approximation coefficients $A_J$ and detail coefficients $D_j$ for scales $j=1,\ldots,J$.
2. Compute subband energies $E_j = \sum_n |D_j[n]|^2$ and optionally include $\|A_J\|_2^2$.
3. Form the feature vector $x_F=[E_1,\ldots,E_J,\|A_J\|_2^2]$.

Using stationary or energy-pooled coefficients provides approximate shift invariance [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- G. Tzanetakis, G. Essl, and P. Cook, "Audio analysis using the discrete wavelet transform," in *Proc. WSES Int. Conf. Acoustics & Music Theory & Applications*, Skiathos, Greece (2001).
