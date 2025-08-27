# MTP Adapter

The Matched-Template Projection (MTP) adapter projects the waveform onto a known canonical template and reports the best-fit amplitude or coefficients. It is included in the first layer of cycle-synchronous mappings [Plan].

## Algorithm
1. Assume a reference pattern $z(t)$ shared across files.
2. Estimate amplitude $a$ and shift $\tau$ by solving
   $$ (\hat a, \hat\tau) = \arg\min_{a,\tau} \sum_n \left(v_{F,n} - a\, z(t_{F,n}-\tau)\right)^2 $$
3. Use $\hat a$ or projection coefficients $\langle v_F, z_k \rangle$ as features $x_F$.

Encoding only amplitudes yields shift-invariant, compact descriptors when the template is accurate [Theory].

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- G. L. Turin, "An introduction to matched filters," *IRE Trans. Information Theory* 6(3), 311–329 (1960).
