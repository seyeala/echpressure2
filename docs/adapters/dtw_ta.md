# DTW-TA Adapter

The Dynamic Time Warping Template Average (DTW-TA) adapter aligns each detected cycle of the waveform to a reference template using constrained DTW and averages the aligned cycles. It is a first-layer cycle-synchronous mapping [Plan].

## Algorithm
1. Build an initial template from a few cycles or another adapter output.
2. Align each cycle to the template via constrained DTW (e.g., Sakoe–Chiba band) and resample to a common length $M$.
3. Robust-average the aligned cycles to produce $x_F \in \mathbb{R}^M$.

This procedure handles cycle length drift and shape variability while preserving shift invariance through alignment [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," *IEEE Trans. Acoust. Speech Signal Process.* 26(1), 43–49 (1978).
