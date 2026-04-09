# Configuration

All simulation parameters are set through frozen dataclasses. Once created, they're immutable — no accidental mutations during a run.

## SimulationConfig

The top-level config that controls everything:

```python
from mcising import (
    SimulationConfig, LatticeConfig, AdaptiveConfig,
    Algorithm, ExecutionMode, LatticeType,
)

config = SimulationConfig(
    lattice=LatticeConfig(
        lattice_type=LatticeType.SQUARE,  # square, triangular, honeycomb, cubic, chain
        size=32,                           # linear extent L
        j1=1.0,                            # nearest-neighbor coupling
        j2=0.0,                            # next-nearest-neighbor coupling
        j3=0.0,                            # third-nearest-neighbor coupling
        h=0.0,                             # external magnetic field
    ),
    algorithm=Algorithm.METROPOLIS,        # metropolis, wolff, swendsen_wang
    seed=42,                               # deterministic RNG seed
    temperatures=(3.0, 2.269, 1.5),        # temperature points
    n_sweeps=1000,                         # measurement sweeps per temperature
    n_thermalization=100,                  # warmup sweeps
    measurement_interval=10,               # measure every N sweeps
    compute_correlation=False,             # compute C(r) per temperature
    adaptive=AdaptiveConfig(enabled=False),# adaptive thermalization
    mode=ExecutionMode.COOLDOWN,           # cooldown, independent, parallel_tempering
    swap_interval=1,                       # sweeps between PT swaps
)
```

## LatticeConfig

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lattice_type` | `LatticeType` | `SQUARE` | Lattice geometry |
| `size` | `int` | `10` | Linear extent L (must be >= 2) |
| `j1` | `float` | `1.0` | Nearest-neighbor coupling |
| `j2` | `float` | `0.0` | Next-nearest-neighbor coupling |
| `j3` | `float` | `0.0` | Third-nearest-neighbor coupling |
| `h` | `float` | `0.0` | External magnetic field |

All coupling values must be finite. Cluster algorithms (Wolff, Swendsen-Wang) require `j2=0`, `j3=0`, `h=0`.

## Algorithm constraints

| Algorithm | J1 | J2 | J3 | h | All lattices? |
|---|---|---|---|---|---|
| Metropolis | any | any | any | any | Yes |
| Wolff | any | 0 only | 0 only | 0 only | Yes |
| Swendsen-Wang | any | 0 only | 0 only | 0 only | Yes |

## Temperature specification

Temperatures must be positive and finite. For the cooldown mode, they're automatically sorted in descending order.

```python
# Individual temperatures
temperatures=(3.0, 2.269, 1.5)

# Dense scan (via numpy)
import numpy as np
temperatures=tuple(np.linspace(1.5, 3.5, 50))
```

## Defaults

| Parameter | Default |
|---|---|
| `lattice_type` | `SQUARE` |
| `size` | `10` |
| `j1` | `1.0` |
| `j2, j3, h` | `0.0` |
| `algorithm` | `METROPOLIS` |
| `seed` | `42` |
| `n_sweeps` | `1000` |
| `n_thermalization` | `100` |
| `measurement_interval` | `10` |
| `mode` | `COOLDOWN` |
| `swap_interval` | `1` |

See the [API Reference](../reference/config.md) for complete documentation of all parameters.
