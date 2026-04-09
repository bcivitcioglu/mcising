# CLI Reference

mcising provides a command-line interface for running simulations and benchmarks without writing Python code.

## `mcising run`

Run a Monte Carlo simulation.

```bash
# Basic run with output
mcising run -L 32 --seed 42 -o results.h5

# Specify temperatures
mcising run -L 32 -T 3.0 -T 2.269 -T 1.5

# Temperature range (start:stop:step)
mcising run -L 32 --T-range 4.0:1.0:0.1

# Choose lattice and algorithm
mcising run -L 32 --lattice triangular --algorithm wolff

# Parallel execution
mcising run -L 32 --mode independent -T 3.0 -T 2.269 -T 1.5

# Parallel tempering
mcising run -L 32 --mode parallel_tempering -T 3.0 -T 2.5 -T 2.0 -T 1.5

# Adaptive mode
mcising run -L 64 --adaptive --min-samples 200

# Checkpointing
mcising run -L 32 --checkpoint sim.h5
mcising run -L 32 --checkpoint sim.h5 --resume

# J1-J2 frustrated model
mcising run -L 32 --j1 1.0 --j2 0.5

# Full options
mcising run -L 32 --lattice honeycomb --j1 1.0 --j3 0.3 \
    --mode independent --algorithm metropolis \
    --sweeps 5000 --therm 500 --seed 42 -o results.h5
```

## `mcising benchmark`

Benchmark mcising performance across all lattices, algorithms, and coupling strategies.

```bash
# Full benchmark report
mcising benchmark

# Custom lattice size and sweep count
mcising benchmark -L 64 --sweeps 50000

# Scaling benchmark across lattice sizes
mcising benchmark --scaling
```

## `mcising info`

Show version and available features.

```bash
mcising info
```
