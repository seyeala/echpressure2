# HTE Adapter

The Hilbert Transform Envelope (HTE) adapter computes the analytic signal and extracts its amplitude envelope, yielding a shift-invariant representation of modulation patterns. It is part of the second-layer transform adapters [Plan].

## Algorithm
1. Form the analytic signal $a(t)=v(t)+i\,\mathcal{H}\{v(t)\}$ using the Hilbert transform $\mathcal{H}\{\cdot\}$.
2. Take the envelope $e(t)=|a(t)|$ and optionally rotate or summarize it to obtain a fixed-length vector $x_F$.

Because the envelope discards the carrier phase, it is insensitive to global time shifts [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- S. O. Sadjadi and J. H. L. Hansen, "Hilbert envelope based features for robust speaker identification under reverberant mismatched conditions," in *Proc. IEEE ICASSP*, 5448–5451 (2011).
