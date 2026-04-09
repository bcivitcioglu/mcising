# Cluster Algorithms

Metropolis flips one spin at a time. Near the critical temperature, this gets slow — the system needs exponentially many flips to decorrelate. Cluster algorithms fix this by flipping entire groups of spins at once.

## The three algorithms

=== "Metropolis"

    ```python
    from mcising import Simulation, SimulationConfig, LatticeConfig, Algorithm

    config = SimulationConfig(
        lattice=LatticeConfig(size=32),
        algorithm=Algorithm.METROPOLIS,
        temperatures=(2.269,),
        n_sweeps=1000,
    )

    results = Simulation(config).run()
    ```

    Single-spin-flip with lookup tables. Best for general use, especially with J2, J3, or h couplings. Supports all lattice types and coupling combinations.

=== "Wolff"

    ```python
    config = SimulationConfig(
        lattice=LatticeConfig(size=32),
        algorithm=Algorithm.WOLFF,
        temperatures=(2.269,),
        n_sweeps=1000,
    )

    results = Simulation(config).run()
    ```

    Builds a single cluster from a random seed via DFS, then flips it. Dramatically reduces autocorrelation at Tc. One "sweep" = one cluster flip.

=== "Swendsen-Wang"

    ```python
    config = SimulationConfig(
        lattice=LatticeConfig(size=32),
        algorithm=Algorithm.SWENDSEN_WANG,
        temperatures=(2.269,),
        n_sweeps=1000,
    )

    results = Simulation(config).run()
    ```

    Partitions the entire lattice into clusters via bond percolation (Union-Find), then independently flips each cluster with 50% probability. One "sweep" = one full partition + flip.

## When to use which

| Scenario | Best algorithm |
|---|---|
| General purpose, any coupling | **Metropolis** |
| Near Tc, J1-only | **Wolff** (fastest decorrelation) |
| Near Tc, many temperatures | **Swendsen-Wang** (good for parallel tempering) |
| J2 or J3 or h active | **Metropolis** (cluster algorithms require J2=J3=h=0) |

!!! warning "Cluster algorithm constraints"
    Wolff and Swendsen-Wang require `j2=0`, `j3=0`, and `h=0`. This is a fundamental limitation of the bond-percolation approach — it only works for pure nearest-neighbor ferromagnets. If you need J2 or external field, use Metropolis.

## Performance comparison

On a 32x32 square lattice at Tc=2.269, 10,000 sweeps:

| Algorithm | Updates/sec |
|---|---|
| Metropolis | 268M |
| Wolff | 100M |
| Swendsen-Wang | 48M |

Metropolis has the highest raw throughput, but Wolff and Swendsen-Wang produce statistically independent samples much faster near Tc because each sweep decorrelates more effectively.

## Cluster algorithms on any lattice

All three algorithms work on all 5 lattice types:

```python
from mcising import LatticeType

# Wolff on triangular lattice
config = SimulationConfig(
    lattice=LatticeConfig(
        lattice_type=LatticeType.TRIANGULAR,
        size=32,
    ),
    algorithm=Algorithm.WOLFF,
    temperatures=(3.641,),
    n_sweeps=1000,
)
```
