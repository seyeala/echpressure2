# PB-CSA Adapter

The Phase-Bin Cycle-Synchronous Averaging (PB-CSA) adapter is a first-layer, cycle-synchronous mapping that folds each waveform by phase and averages across cycles to form a fixed-length vector [Plan].

## Algorithm
1. Estimate the fundamental period $T_0$ and compute sample phases
   $\theta_n = 2\pi (t_{F,n} \bmod T_0)/T_0$ [Theory].
2. Partition the phase domain into $K$ bins $\Theta_k$ and aggregate samples in each bin using a robust statistic such as the mean or median [Theory].
3. Circularly shift the resulting vector so a reference phase is centered, ensuring approximate shift invariance [Theory].

The output is $x_F \in \mathbb{R}^K$, where $K$ is the number of phase bins.

## References
- **Plan:** *Modular Python Repository Architecture for Pressure–Oscilloscope Dataset Processing, Alignment, Adapters, and Visualization*.
- **Theory:** *Raw Dataset Specification: Pressure & Oscilloscope Streams with File-Level Midpoint Alignment and Configurable Uncertainty*.
- S. Braun, "The Extraction of Periodic Waveforms by Time Domain Averaging," *Acustica* 32(2), 69–77 (1975).
