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

Independent mode runs one temperature per Rayon thread; parallel tempering
does the same and adds the replica-exchange synchronisation. The cooldown
mode is the single-threaded reference.

<!-- benchmarks:parallel:begin -->
| Mode | Threads | Wall time | Speed-up vs cooldown |
|---|---|---|---|
| Cooldown | 1 | 3.01 s | 1.0× |
| Independent | 1 | 3.03 s | 1.0× |
| Independent | 2 | 1.62 s | 1.9× |
| Independent | 4 | 0.84 s | 3.6× |
| Independent | 8 | 0.57 s | 5.3× |
| Independent | 10 | 0.54 s | 5.6× |
| Parallel tempering | 10 | 1.47 s | 2.0× |

Metropolis, 128×128 square lattice, 20 temperatures from 3.5 to 1.5, 500 thermalization + 2,000 production sweeps per temperature (measured every 10), Apple M4 (10 cores: 4 performance + 6 efficiency); medians of 3 runs. The independent and parallel-tempering rows each run in a fresh process with `RAYON_NUM_THREADS` set to the thread count; the cooldown mode is single-threaded by construction.
<!-- benchmarks:parallel:end -->

Independent mode gives the wall-clock speed-up; it grows with the number
of performance cores until the temperatures run out. Parallel tempering
pays for the swap synchronisation on top, in exchange for far better
sampling of the low-temperature replicas — which is what it is for.

## RNG seeding

Each mode handles reproducibility differently:

- **Cooldown**: single seed, single RNG stream
- **Independent**: `seed + temperature_index` per replica — deterministic and
  independent. The index is the temperature's position in the configured
  scan, and it sticks to its temperature even when a checkpointed run
  resumes with some temperatures already done — a resumed scan reproduces
  the uninterrupted run's streams exactly.
- **Parallel Tempering**: per-replica seeds as in independent mode, plus a
  separate swap-decision RNG

All modes are fully deterministic: same seed = same results.

## Checkpointing

All three modes work with `checkpoint_run`. Granularity differs: cooldown
saves after each temperature, independent mode saves the whole batch when
it returns (resume runs only the missing temperatures), and parallel
tempering is all-or-nothing because the replicas form one coupled
ensemble. See [Saving results](../guide/saving-results.md#checkpointing-crash-recovery).

## CLI

```bash
# Default (cooldown)
mcising run -L 32 -T 3.0 -T 2.269 -T 1.5

# Independent parallel
mcising run -L 32 --mode independent -T 3.0 -T 2.269 -T 1.5

# Parallel tempering
mcising run -L 32 --mode parallel_tempering -T 3.0 -T 2.269 -T 1.5
```
