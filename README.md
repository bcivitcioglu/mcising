<p align="center">
  <img src="assets/logo.svg" alt="mcising logo" width="300">
</p>

<h1 align="center">mcising</h1>

<p align="center">
  High-performance Ising model Monte Carlo simulation with a Rust core.
</p>

<p align="center">
  <a href="https://pepy.tech/project/mcising"><img src="https://static.pepy.tech/badge/mcising" alt="Downloads"></a>
</p>

---

**mcising** is a Python library for Monte Carlo simulation of Ising spin systems. It supports 5 lattice geometries, J1-J2-J3 frustrated magnetism with external fields, 3 Monte Carlo algorithms, 3 execution modes (including parallel tempering), and adaptive thermalization. The performance-critical core is written in Rust via PyO3.

## Performance

<!-- benchmarks:headline:begin -->
On one core of an Apple M4 (10 cores: 4 performance + 6 efficiency), mcising performs **351M Metropolis spin updates per second** on a 32×32 square lattice at Tc — 140.4× faster than pure Python and 15.0× faster than a NumPy checkerboard implementation of the same update, and 2.4× faster than peapods on a matched workload (energy recorded every sweep on both sides).

mcising 0.29.0 (commit 2e3548a), Python 3.12.11, measured 2026-09-01; medians of repeated runs. Regenerate with `uv run --group benchmark python benchmarks/run_all.py --write-docs`.
<!-- benchmarks:headline:end -->

<!-- benchmarks:baselines:begin -->
| Implementation | Lattice | Timed sweeps | Spin updates/s | mcising is |
|---|---|---|---|---|
| Pure Python | 32×32 | 200 | 2M | 140.4× |
| NumPy (checkerboard) | 32×32 | 1,000 | 23M | 15.0× |
| mcising (Rust) | 32×32 | 10,000 | 351M | — |
| NumPy (checkerboard) | 128×128 | 200 | 69M | 4.7× |
| mcising (Rust) | 128×128 | 1,000 | 324M | — |

Single-spin-flip Metropolis on the square lattice at Tc, one thread, Apple M4 (10 cores: 4 performance + 6 efficiency). Pure Python is a plain loop with precomputed neighbour and Boltzmann tables; the NumPy implementation updates the two checkerboard sublattices as whole arrays. Spin updates/s counts attempted flips; timed sweeps exclude 100 warm-up sweeps.
<!-- benchmarks:baselines:end -->

<!-- benchmarks:peapods:begin -->
Matched physics against [peapods](https://github.com/PeaBrane/peapods) 0.2.0 (Rust/PyO3): the same Hamiltonian (H = −J Σ s_i s_j, J = 1, energies per site, bond counted once), the same Metropolis sweep (a sequential scan with one attempt per site), the same temperature (Tc), 5,000 thermalization sweeps, then 100,000 timed sweeps with the energy recorded every sweep on both sides, one thread, 3 seeds per side, Apple M4 (10 cores: 4 performance + 6 efficiency). peapods reports +Σ J s_i s_j / N, so its sign is flipped before comparison. A row is published only when the two mean energies agree within 0.5 %.

| Lattice | E/site mcising | E/site peapods | Δ | Sweeps/s mcising | Sweeps/s peapods | mcising is |
|---|---|---|---|---|---|---|
| Square 32×32 | -1.4325 ± 0.0018 | -1.4346 ± 0.0007 | 0.15 % | 249,160 | 104,786 | 2.4× |
| Triangular 32×32 | -2.0305 ± 0.0012 | -2.0320 ± 0.0014 | 0.07 % | 203,737 | 86,013 | 2.4× |
| Cubic 16×16×16 | -1.0346 ± 0.0010 | -1.0347 ± 0.0002 | 0.01 % | 34,196 | 15,677 | 2.2× |

Wolff and Swendsen-Wang are not compared: peapods interleaves cluster updates with Metropolis sweeps rather than running cluster-only sweeps, so no peapods workload matches an mcising cluster sweep.
<!-- benchmarks:peapods:end -->

Every number above is written by [`benchmarks/run_all.py`](benchmarks/run_all.py) from the committed [`benchmarks/results.json`](benchmarks/results.json), and the test suite fails if the two drift apart. Regenerate with `uv run --group benchmark python benchmarks/run_all.py --write-docs` (a few minutes), or re-render the committed results with `--from-json benchmarks/results.json --write-docs`. The [performance page](https://bcivitcioglu.github.io/mcising/advanced/performance/) adds the per-lattice, cluster-algorithm, scaling and parallel-execution tables.

mcising also supports features not available in peapods: J2/J3 coupling, external magnetic field, honeycomb lattice, 1D chain, and parallel tempering.

## Features

- **5 lattice geometries** -- square, triangular, honeycomb (2-sublattice), cubic (3D), chain (1D)
- **3 MC algorithms** -- Metropolis, Wolff cluster, Swendsen-Wang cluster
- **3 execution modes** -- sequential cool-down, independent parallel (Rayon), parallel tempering with replica exchange
- **J1-J2-J3 frustrated magnetism** -- nearest, next-nearest, and third-nearest-neighbor couplings
- **External magnetic field** -- h coupling, compatible with all lattices
- **15 Metropolis strategies** -- auto-selected lookup tables optimized per coupling combination
- **Adaptive thermalization** -- MSER equilibration detection + Sokal autocorrelation estimation
- **Correlation functions** -- spin-spin correlation and correlation length
- **HDF5 output** with crash-safe incremental checkpointing
- **Rich CLI** with progress bars, benchmarking, and structured output
- **Fully reproducible** -- deterministic RNG (Xoshiro256**), same seed = same results

## Installation

```bash
pip install mcising

# with plotting (matplotlib) and/or DataFrame export (pandas):
pip install 'mcising[plot]'
pip install 'mcising[plot,dataframe]'
```

For development (requires Rust toolchain):

```bash
git clone https://github.com/bcivitcioglu/mcising.git
cd mcising
uv sync
uv run maturin develop
```

## Quick Start

### Python API

```python
from mcising import Simulation, SimulationConfig, LatticeConfig, LatticeType

config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    seed=42,
)

sim = Simulation(config)
results = sim.run()

# Access results per temperature
for T in results.temperatures:
    print(f"T={T:.3f}: <E>={results.energy[T].mean():.4f}, "
          f"<|M|>={abs(results.magnetization[T]).mean():.4f}")
```

### Multiple Lattice Types

```python
from mcising import LatticeType

# Triangular lattice with J1-J2 frustration
config = SimulationConfig(
    lattice=LatticeConfig(
        lattice_type=LatticeType.TRIANGULAR,
        size=32,
        j1=1.0,
        j2=0.5,
    ),
    temperatures=(4.0, 3.641, 2.0),
    n_sweeps=1000,
)

# Also available: HONEYCOMB, CUBIC, CHAIN
```

### Parallel Execution

```python
from mcising import ExecutionMode

# Independent: each temperature runs in parallel (uses all CPU cores)
config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(3.0, 2.5, 2.269, 2.0, 1.5),
    n_sweeps=1000,
    mode=ExecutionMode.INDEPENDENT,  # one temperature per core
)

# Parallel Tempering: parallel + replica swap for better sampling
config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(3.0, 2.5, 2.269, 2.0, 1.5),
    n_sweeps=1000,
    mode=ExecutionMode.PARALLEL_TEMPERING,
)
```

### Adaptive Mode

For large lattices near the critical temperature, enable adaptive measurement to automatically determine thermalization length and measurement spacing:

```python
from mcising import AdaptiveConfig

config = SimulationConfig(
    lattice=LatticeConfig(size=64),
    temperatures=(3.0, 2.269, 1.5),
    adaptive=AdaptiveConfig(enabled=True, min_independent_samples=200),
    seed=42,
)

results = Simulation(config).run()

# Inspect diagnostics
for T in results.temperatures:
    diag = results.adaptive_diagnostics[T]
    print(f"T={T:.3f}: tau_int={diag.tau_int:.1f}, "
          f"interval={diag.measurement_interval}")
```

### CLI

```bash
# Basic run
mcising run -L 32 --seed 42 -o results.h5

# Triangular lattice with parallel tempering
mcising run -L 32 --lattice triangular --mode parallel_tempering

# Independent parallel execution (uses all CPU cores)
mcising run -L 32 --mode independent -T 3.0 -T 2.269 -T 1.5

# Adaptive mode
mcising run -L 64 --adaptive --min-samples 200 --seed 42

# With checkpointing (crash-safe)
mcising run -L 32 --checkpoint sim.h5

# Resume interrupted run
mcising run -L 32 --checkpoint sim.h5 --resume

# Benchmark performance across all lattices and algorithms
mcising benchmark

# Show info
mcising info
```

### Saving Results

```python
from mcising import save_hdf5, load_hdf5, save_json_summary

# HDF5 (full data)
save_hdf5(results, "results.h5")
loaded = load_hdf5("results.h5")

# JSON summary (statistics only)
save_json_summary(results, "summary.json")
```

## Architecture

```
mcising/
├── rust/src/              # Rust core (compiled to mcising._core)
│   ├── algorithm/         # MC algorithms (Metropolis, Wolff, Swendsen-Wang)
│   ├── autocorrelation.rs # MSER + Sokal windowing
│   ├── lattice/           # Lattice geometries (square, triangular, honeycomb, cubic, chain)
│   ├── observables.rs     # Energy, magnetization, correlation
│   ├── parallel.rs        # Rayon-parallelized execution (independent + parallel tempering)
│   └── simulation.rs      # PyO3 boundary (IsingSimulation)
├── python/mcising/        # Python package
│   ├── simulation.py      # High-level Simulation class
│   ├── config.py          # Frozen dataclass configs
│   ├── io.py              # HDF5/JSON I/O
│   ├── plotting.py        # Matplotlib visualization
│   └── cli.py             # Typer CLI
├── tests/                 # 401 tests (141 Rust + 260 Python)
└── benchmarks/            # Reproducible performance comparisons
```

## License

This project is licensed under the MIT License.
