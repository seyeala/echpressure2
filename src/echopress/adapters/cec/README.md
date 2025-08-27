# CEC Adapter

The Cepstral Envelope Coefficients (CEC) adapter is a first-layer method that captures the spectral envelope of a periodic signal using low-quefrency cepstral coefficients [Plan].

## Algorithm
1. Compute the log-magnitude spectrum of the waveform and take its inverse Fourier transform to obtain the real cepstrum $C(q)$.
2. Collect the first $Q$ low-quefrency coefficients $c_q$ as the feature vector $x_F=[c_1,\ldots,c_Q]$.
3. Because coefficients depend only on the magnitude spectrum, the representation is shift-invariant and robust to noise [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- A. M. Noll, "Cepstrum pitch determination," *J. Acoust. Soc. Am.* 41(2), 293–309 (1967).
