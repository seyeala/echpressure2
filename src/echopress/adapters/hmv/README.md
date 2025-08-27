# HMV Adapter

The Harmonic Magnitude Vector (HMV) adapter is a first-layer cycle-synchronous method that summarizes a signal by the magnitudes of its first $H$ harmonics of the fundamental frequency [Plan].

## Algorithm
1. Estimate $f_0$ and compute
   $$X(k)=\sum_n v_{F,n} e^{-i2\pi k f_0 t_{F,n}},\quad k=1,\ldots,H$$
   where $v_{F,n}$ are samples of the waveform.
2. Form the feature vector $x_F=[|X(1)|,\ldots,|X(H)|]$.

Magnitudes remove phase, yielding shift invariance [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, Prentice Hall (1989).
