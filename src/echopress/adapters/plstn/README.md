# PLSTN Adapter

The Peak-Locked Segmentation & Time Normalization (PLSTN) adapter is part of the first layer of cycle-synchronous, shift-invariant mappings [Plan]. It centers windows on detected peaks, normalizes each cycle to a common length, and aggregates them.

## Algorithm
1. Detect peak indices $\{n_j\}$ with a refractory period near $T_0$.
2. For each cycle window $W_j=[n_j-w_-, n_j+w_+]$, resample to $M$ points using bandlimited or cubic interpolation.
3. Robust-average the normalized cycles to obtain $x_F \in \mathbb{R}^M$.

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- P. D. McFadden, "Interpolation techniques for time domain averaging of gear vibration," *Mechanical Systems and Signal Processing* 3(1), 87–97 (1989).
