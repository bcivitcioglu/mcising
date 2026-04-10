# CLI Reference

mcising provides a full command-line interface for running simulations, inspecting results, generating plots, and exporting data — all without writing Python code.

## Overview

```bash
mcising run        # Run a simulation → HDF5 file
mcising summary    # Inspect results from HDF5
mcising plot       # Generate plots from HDF5
mcising export     # Export lattice PNGs to zip
mcising benchmark  # Performance benchmark
mcising docs       # Capability reference (agent-readable)
mcising info       # Version and build info
```

## `mcising run`

Run a Monte Carlo simulation and save results to HDF5.

```bash
# Basic run
mcising run -L 32 --seed 42 -o results.h5

# Specify temperatures
mcising run -L 32 -T 3.0 -T 2.269 -T 1.5 -o results.h5

# Temperature range (start:stop:step)
mcising run -L 32 --T-range 4.0:1.0:0.1 -o results.h5

# Choose lattice and algorithm
mcising run -L 32 --lattice triangular --algorithm wolff -o results.h5

# J1-J2 frustrated model
mcising run -L 32 --j1 1.0 --j2 0.5 -o results.h5

# Parallel execution (uses all CPU cores)
mcising run -L 32 --mode independent -T 3.0 -T 2.269 -T 1.5 -o results.h5

# Parallel tempering
mcising run -L 32 --mode parallel_tempering -T 3.0 -T 2.5 -T 2.0 -T 1.5 -o results.h5

# Adaptive mode
mcising run -L 64 --adaptive --min-samples 200 -o results.h5

# Checkpointing (crash-safe)
mcising run -L 32 --checkpoint sim.h5
mcising run -L 32 --checkpoint sim.h5 --resume

# Full combo
mcising run -L 32 --lattice triangular --j1 1.0 --j2 0.5 \
    --algorithm metropolis --mode independent \
    --adaptive --seed 42 -o results.h5
```

## `mcising summary`

Inspect simulation results from an HDF5 file. Shows mean energy, magnetization, specific heat, and susceptibility per temperature.

```bash
# Rich table (default)
mcising summary results.h5

# JSON output (for agents/scripts)
mcising summary results.h5 --json

# CSV output
mcising summary results.h5 --csv
```

Example output:

```
                    Simulation Results
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃      T ┃   <E>/N ┃ <|M|>/N ┃   Cv/N ┃    chi/N ┃ samples ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ 1.5000 │ -1.9523 │  0.9868 │ 0.2210 │   0.0305 │     200 │
│ 2.2690 │ -1.4144 │  0.6503 │ 1.8544 │ 195.7559 │     200 │
│ 3.5000 │ -0.6630 │  0.0605 │ 0.2657 │   1.5562 │     200 │
└────────┴─────────┴─────────┴────────┴──────────┴─────────┘
```

## `mcising plot`

Generate plots from HDF5 results. Output file (`-o`) is always required.

### Thermodynamic quantities (vs temperature)

```bash
mcising plot energy results.h5 -o energy.png
mcising plot magnetization results.h5 -o mag.png
mcising plot specific-heat results.h5 -o cv.png
mcising plot susceptibility results.h5 -o chi.png
```

Multi-file overlay for comparing different coupling configurations:

```bash
mcising plot energy j2_0.h5 j2_0.3.h5 j2_0.5.h5 -o compare.png
```

### Lattice configurations

```bash
# All configs at T=2.269 side by side
mcising plot lattice results.h5 -o lattice.png -T 2.269

# Single config (#5)
mcising plot lattice results.h5 -o lattice_5.png -T 2.269 --n 5
```

### Diagnostics

```bash
# Energy time series (check thermalization)
mcising plot timeseries results.h5 -o trace.png -T 2.269

# Magnetization histogram (bimodal below Tc)
mcising plot histogram results.h5 -o hist.png -T 2.269
```

All plot commands accept `--dpi` (default 150).

## `mcising export`

Export every lattice configuration as a PNG image in a zip file. Filenames encode lattice type, size, couplings, algorithm, temperature, and config number.

```bash
# Tree structure (folders per temperature)
mcising export results.h5 lattices.zip

# Flat structure (all PNGs in one folder)
mcising export results.h5 lattices.zip --flat

# Export only specific temperatures
mcising export results.h5 lattices.zip -T 2.269 -T 1.5
```

## `mcising benchmark`

Benchmark mcising performance across all lattices, algorithms, and coupling strategies.

```bash
mcising benchmark
mcising benchmark -L 64 --sweeps 50000
mcising benchmark --scaling
```

## `mcising docs`

Machine-readable capability reference. Designed for AI agents to discover what mcising can do.

```bash
mcising docs              # Full CLI reference with examples
mcising docs lattices     # Lattice types + Tc + coordination
mcising docs algorithms   # Algorithms + constraints
mcising docs couplings    # J1/J2/J3/H support per lattice
mcising docs modes        # Execution modes
mcising docs cli          # All commands with examples
```

## `mcising info`

Display version, build info, and available algorithms.

```bash
mcising info
```
