# Parallel Execution

mcising offers three execution modes for temperature scans. The right choice depends on your use case.

## The three modes

### Cooldown (default)

Temperatures are processed sequentially from high to low. The spin configuration carries forward — each temperature starts from the previous one's final state.

```python
from mcising import Simulation, SimulationConfig, LatticeConfig, ExecutionMode

config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(3.0, 2.5, 2.269, 2.0, 1.5),
    n_sweeps=1000,
    mode=ExecutionMode.COOLDOWN,  # this is the default
)

results = Simulation(config).run()
```

Best for: avoiding metastable states at low temperature. Single-threaded.

### Independent

Each temperature runs from random initialization on a separate CPU core. No communication between temperatures.

```python
config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(3.0, 2.5, 2.269, 2.0, 1.5),
    n_sweeps=1000,
    mode=ExecutionMode.INDEPENDENT,
)

results = Simulation(config).run()
```

Best for: fast scans with many temperatures. Uses all CPU cores via Rayon.

### Parallel Tempering

Like independent mode, but replicas periodically **swap** spin configurations between adjacent temperatures. High-temperature replicas explore freely and pass good configurations down to low-temperature replicas.

```python
config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(3.0, 2.5, 2.269, 2.0, 1.5),
    n_sweeps=1000,
    mode=ExecutionMode.PARALLEL_TEMPERING,
    swap_interval=1,  # attempt swap every sweep (default)
)

results = Simulation(config).run()
```

Best for: frustrated systems, spin glasses, any system where Metropolis gets stuck in local minima.

## When to use which

| Mode | Parallelism | Sampling quality | Use case |
|---|---|---|---|
| **Cooldown** | None (1 core) | Good (warm start) | Standard simulations |
| **Independent** | Full (all cores) | OK (cold start per T) | Fast scans, error bars |
| **Parallel Tempering** | Full + swap sync | Best (replica exchange) | Frustrated/glassy systems |

## Speed comparison

With 20 temperatures on a 10-core machine:

| Mode | Wall-clock time | Speedup |
|---|---|---|
| Cooldown | 0.018s | 1.0x |
| Independent | 0.003s | **6x faster** |
| Parallel Tempering | 0.075s | slower (swap overhead) |

Independent mode gives the biggest wall-clock speedup. Parallel Tempering is slower in raw time but produces better-sampled configurations at low temperature.

## RNG seeding

Each mode handles reproducibility differently:

- **Cooldown**: single seed, single RNG stream
- **Independent**: `seed + temperature_index` per replica — deterministic and independent
- **Parallel Tempering**: same as independent, plus a separate swap RNG

All modes are fully deterministic: same seed = same results.

## CLI

```bash
# Default (cooldown)
mcising run -L 32 -T 3.0 -T 2.269 -T 1.5

# Independent parallel
mcising run -L 32 --mode independent -T 3.0 -T 2.269 -T 1.5

# Parallel tempering
mcising run -L 32 --mode parallel_tempering -T 3.0 -T 2.269 -T 1.5
```
