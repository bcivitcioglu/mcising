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

On a 32x32 square lattice at Tc=2.269, 10,000 sweeps:

| Algorithm | Updates/sec |
|---|---|
| Metropolis | 268M |
| Wolff | 100M[^wolff-updates] |
| Swendsen-Wang | 48M |

[^wolff-updates]: Measured before 1.0, when the benchmark counted one
    Wolff sweep as `num_sites` updates — an overstatement of roughly
    `N / ⟨cluster size⟩` (a Wolff sweep is one cluster). The benchmark
    now counts real attempted flips; re-measured numbers land with the
    benchmark-integrity pass (P17).

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
