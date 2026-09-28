# DTW-TA Adapter

**Status: basic cycle segmentation is implemented; DTW template alignment and
averaging are planned.** The existing adapter identifier is `dtw_ta`.

## Current implementation

`DtwTaAdapter.layer1` calls `cycle_synchronous_map`. It splits a
one-dimensional signal from its first sample into consecutive fixed-length
rows. The cycle length is `int(fs / f0)`, unless the `TARGET_CYCLE_LEN`
environment variable overrides it. A trailing incomplete cycle is discarded.

`layer2` returns `{"cycles": cycles}` without further processing. The output
has shape `(n_cycles, cycle_len)`. The adapter does not currently detect cycle
anchors, construct a template, compute a DTW path, resample aligned cycles,
or average them. It does not provide a shift-invariance guarantee.

## Planned DTW-TA algorithm

The Dynamic Time Warping Template Average design in the roadmap proposes:

1. Building an initial reference template from selected cycles.
2. Aligning cycles to it with constrained DTW, such as a Sakoe–Chiba band.
3. Resampling aligned cycles to a common length and computing a robust average.

These steps describe planned work, not behavior available by selecting
`--adapter dtw_ta`. The current segmented output must not be interpreted as a
DTW-aligned template average.

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," *IEEE Trans. Acoust. Speech Signal Process.* 26(1), 43–49 (1978).
