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

**mcising** is a Python library for Monte Carlo simulation of Ising spin systems on square lattices. It supports nearest-neighbor (J1) and next-nearest-neighbor (J2) interactions, external magnetic fields, correlation functions, and adaptive thermalization. The performance-critical simulation loop is written in Rust via PyO3.

## Features

- **Rust-accelerated Metropolis algorithm** via PyO3/maturin
- **J1-J2 frustrated magnetism** -- nearest and next-nearest-neighbor couplings
- **Adaptive thermalization** -- MSER equilibration detection + Sokal autocorrelation estimation
- **Cool-down approach** -- temperatures processed in descending order to avoid metastable states
- **Correlation functions** -- spin-spin correlation and correlation length
- **HDF5 output** with crash-safe incremental checkpointing
- **Rich CLI** with progress bars, benchmarking, and structured output
- **Fully reproducible** -- deterministic RNG (Xoshiro256**), same seed = same results

## Installation

```bash
pip install mcising
```

For development (requires Rust toolchain):

```bash
git clone https://github.com/burakericok/mcising.git
cd mcising
uv sync
uv run maturin develop
```

## Quick Start

### Python API

```python
from mcising import Simulation, SimulationConfig, LatticeConfig

config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0, j2=0.0),
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

# Adaptive mode
mcising run -L 64 --adaptive --min-samples 200 --seed 42

# With checkpointing (crash-safe)
mcising run -L 32 --checkpoint sim.h5

# Resume interrupted run
mcising run -L 32 --checkpoint sim.h5 --resume

# Benchmark performance
mcising benchmark -L 64 --sweeps 10000

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
│   ├── algorithm/         # MC algorithms (Metropolis)
│   ├── autocorrelation.rs # MSER + Sokal windowing
│   ├── lattice/           # Lattice geometries (square)
│   ├── observables.rs     # Energy, magnetization, correlation
│   └── simulation.rs      # PyO3 boundary (IsingSimulation)
├── python/mcising/        # Python package
│   ├── simulation.py      # High-level Simulation class
│   ├── config.py          # Frozen dataclass configs
│   ├── io.py              # HDF5/JSON I/O
│   ├── plotting.py        # Matplotlib visualization
│   └── cli.py             # Typer CLI
└── tests/                 # pytest suite (94 tests)
```

## License

This project is licensed under the MIT License.