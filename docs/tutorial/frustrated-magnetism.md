# Frustrated Magnetism

When you add competing interactions to the Ising model, spins can't satisfy all their neighbors at once. This is **frustration** — and it leads to rich physics that mcising is built to explore.

## J1-only: the clean case

With only nearest-neighbor coupling, the ground state is simple — all spins align:

```python
from mcising import Simulation, SimulationConfig, LatticeConfig

config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    seed=42,
)

results = Simulation(config).run()
```

## Adding J2: next-nearest-neighbor coupling

J2 couples diagonal neighbors. When J2 competes with J1, the system becomes frustrated:

```python
config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0, j2=0.5),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    seed=42,
)

results = Simulation(config).run()

for T in results.temperatures:
    E = results.energy[T].mean()
    M = abs(results.magnetization[T]).mean()
    print(f"T={T:.3f}: <E>={E:.4f}, <|M|>={M:.4f}")
```

!!! info "What J2 does"
    Positive J2 favors aligned diagonal neighbors (reinforces J1). Negative J2 favors anti-aligned diagonals (competes with J1). The ratio J2/J1 controls the degree of frustration.

## J1-J2-J3: the full Hamiltonian

Add third-nearest-neighbor coupling for even richer phase behavior:

```python
config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0, j2=0.5, j3=0.3),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    seed=42,
)

results = Simulation(config).run()
```

mcising automatically selects the optimal Metropolis strategy from 15 pre-built lookup tables based on which couplings are active. No performance cost for adding J2 or J3.

## External magnetic field

Break the up/down symmetry with an external field `h`:

```python
config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0, h=0.5),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    seed=42,
)

results = Simulation(config).run()
```

With h > 0, the system prefers spin-up even above Tc. The phase transition becomes a crossover instead of a sharp transition.

## Frustration on different lattices

Different lattice geometries give different frustration physics:

=== "Triangular J1-J2"

    ```python
    from mcising import LatticeType

    config = SimulationConfig(
        lattice=LatticeConfig(
            lattice_type=LatticeType.TRIANGULAR,
            size=32, j1=1.0, j2=0.5,
        ),
        temperatures=(4.0, 3.641, 2.0),
        n_sweeps=1000,
    )
    ```

    The triangular lattice with antiferromagnetic J1 is geometrically frustrated even without J2.

=== "Honeycomb J1-J2"

    ```python
    config = SimulationConfig(
        lattice=LatticeConfig(
            lattice_type=LatticeType.HONEYCOMB,
            size=32, j1=1.0, j2=0.3,
        ),
        temperatures=(2.0, 1.519, 1.0),
        n_sweeps=1000,
    )
    ```

    J2 on honeycomb couples same-sublattice neighbors (6 per site), competing with the 3 inter-sublattice J1 bonds.

!!! tip
    For frustrated systems near critical points, consider using [Parallel Tempering](parallel-execution.md) — replica swaps help escape local energy minima that trap standard Metropolis.
