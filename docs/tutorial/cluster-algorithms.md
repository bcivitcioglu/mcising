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

    Builds a single cluster from a random seed via DFS, then flips it. Dramatically reduces autocorrelation at Tc.

    !!! warning "Wolff sweeps are cluster updates, not lattice sweeps"
        One Wolff "sweep" = **one cluster flip**, not `num_sites` flip
        attempts like Metropolis or Swendsen-Wang. Away from Tc clusters
        are small, so scale `n_sweeps` up by roughly `N / ⟨cluster size⟩`
        for comparable statistics. `results.n_cluster_flips` records the
        clusters actually flipped per temperature, and each sweep's real
        flip count is in the counters `Simulation.sweep()` returns. (An
        automatic equal-work sweep is deliberately absent: stopping a
        sweep when a flip budget is met biases the sampling — see issue
        #42.)

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

    Partitions the entire lattice into clusters via bond percolation (Union-Find), then independently flips each cluster with 50% probability. One "sweep" = one full partition + flip: every site receives a decision, so a Swendsen-Wang sweep is comparable work to a Metropolis sweep.

## When to use which

| Scenario | Best algorithm |
|---|---|
| General purpose, any coupling | **Metropolis** |
| Near Tc, J1-only | **Wolff** (fastest decorrelation) |
| Near Tc, many temperatures | **Swendsen-Wang** (good for parallel tempering) |
| J2 or J3 or h active | **Metropolis** (cluster algorithms require J2=J3=h=0) |

!!! warning "Cluster algorithm constraints"
    Wolff and Swendsen-Wang require `j2=0`, `j3=0`, and `h=0`. This is a fundamental limitation of the bond-percolation approach — it only works for pure nearest-neighbor ferromagnets. If you need J2 or external field, use Metropolis.

    `j1` must also be positive: the bond probability `1 - exp(-2*beta*J1)` is
    not a probability for antiferromagnetic couplings, so `j1<=0` is rejected
    with a `ConfigurationError`. Use Metropolis for antiferromagnetic
    couplings (a sublattice-mapped cluster algorithm is future work).

## Performance comparison

Raw throughput and statistical efficiency pull in opposite directions.
Metropolis attempts the most flips per second, but near Tc its samples
stay correlated for many sweeps; one Wolff cluster or one Swendsen-Wang
sweep decorrelates the magnetization within a few. The last column — wall
time per statistically independent sample — is the one to compare.

<!-- benchmarks:cluster:begin -->
| Algorithm | µs per sweep | Attempted flips/s | τ_int (energy) | τ_int (abs. magnetization) | µs per independent sample |
|---|---|---|---|---|---|
| Metropolis | 2.93 µs | 349M | 10.8 | 29.9 | 175.54 µs |
| Wolff | 8.10 µs | 59M | 4.3 | 3.9 | 63.15 µs |
| Swendsen-Wang | 16.96 µs | 60M | 4.5 | 3.7 | 124.26 µs |

32×32 square lattice at Tc, one thread, Apple M4 (10 cores: 4 performance + 6 efficiency). Attempted flips/s counts real work: a Metropolis sweep attempts every site once, a Swendsen-Wang sweep touches every site, and one Wolff sweep is one cluster (475 spins on average here). τ_int is the integrated autocorrelation time in sweeps from a 100,000-sweep series after 5,000 thermalization sweeps (blocking estimate). µs per independent sample = µs per sweep × 2 τ_int of the absolute magnetization, the slowest observable.
<!-- benchmarks:cluster:end -->

## Cluster algorithms on any lattice

All three algorithms work on all 5 lattice types:

```python
from mcising import Algorithm, LatticeConfig, LatticeType, SimulationConfig

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
