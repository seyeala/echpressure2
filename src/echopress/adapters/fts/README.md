# FTS Adapter

The Fourier Transform Spectrum (FTS) adapter belongs to the repository's second layer of transform-based mappings, providing a shift-invariant magnitude spectrum representation [Plan].

## Algorithm
Given a time series $v_n$, compute the discrete Fourier transform
$$X(k)=\sum_{n=0}^{M-1} v_n e^{-i2\pi kn/M},\quad k=0,\ldots,M-1.$$
The feature vector consists of the magnitudes $|X(k)|$ (typically the first $M/2+1$ values for real signals). A circular time shift affects only phase, leaving magnitudes unchanged [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, Prentice Hall (1989).
