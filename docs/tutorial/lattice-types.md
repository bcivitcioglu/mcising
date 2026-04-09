# Lattice Types

mcising supports 5 lattice geometries, each with its own coordination number and critical temperature. Let's explore them.

## Available lattices

| Lattice | Dimension | Coordination | Tc (J1=1) | Shape |
|---|---|---|---|---|
| **Square** | 2D | 4 | 2.269 | (L, L) |
| **Triangular** | 2D | 6 | 3.641 | (L, L) |
| **Honeycomb** | 2D | 3 | 1.519 | (L, L, 2) |
| **Cubic** | 3D | 6 | 4.512 | (L, L, L) |
| **Chain** | 1D | 2 | 0 (no transition) | (N,) |

## Choosing a lattice

=== "Square"

    ```python
    from mcising import LatticeConfig, LatticeType

    lattice = LatticeConfig(
        lattice_type=LatticeType.SQUARE,
        size=32,
        j1=1.0,
    )
    ```

    The default. 4 nearest neighbors per site. The classic Ising model that Onsager solved exactly.

=== "Triangular"

    ```python
    from mcising import LatticeConfig, LatticeType

    lattice = LatticeConfig(
        lattice_type=LatticeType.TRIANGULAR,
        size=32,
        j1=1.0,
    )
    ```

    6 nearest neighbors per site. Higher coordination means stronger ordering — Tc is higher than square. Uses offset coordinates internally.

=== "Honeycomb"

    ```python
    from mcising import LatticeConfig, LatticeType

    lattice = LatticeConfig(
        lattice_type=LatticeType.HONEYCOMB,
        size=32,
        j1=1.0,
    )
    ```

    3 nearest neighbors per site. Two-sublattice structure — the spin array has shape `(L, L, 2)`. Lower coordination means weaker ordering (Tc < square). Think graphene geometry.

=== "Cubic"

    ```python
    from mcising import LatticeConfig, LatticeType

    lattice = LatticeConfig(
        lattice_type=LatticeType.CUBIC,
        size=16,
        j1=1.0,
    )
    ```

    3D lattice with 6 nearest neighbors. Spin array has shape `(L, L, L)`. The highest Tc of the group. Use smaller L (16 or less) since the number of sites scales as L^3.

=== "Chain"

    ```python
    from mcising import LatticeConfig, LatticeType

    lattice = LatticeConfig(
        lattice_type=LatticeType.CHAIN,
        size=1000,
        j1=1.0,
    )
    ```

    1D ring with 2 nearest neighbors. No phase transition at any finite temperature (Tc=0). Useful for teaching and testing.

## Running a simulation on any lattice

Once you have a `LatticeConfig`, plug it into `SimulationConfig`:

```python
from mcising import Simulation, SimulationConfig, LatticeConfig, LatticeType

config = SimulationConfig(
    lattice=LatticeConfig(
        lattice_type=LatticeType.TRIANGULAR,
        size=32,
        j1=1.0,
    ),
    temperatures=(4.0, 3.641, 2.0),
    n_sweeps=1000,
    seed=42,
)

results = Simulation(config).run()
```

All algorithms (Metropolis, Wolff, Swendsen-Wang) and execution modes (cooldown, independent, parallel tempering) work on all lattice types.

## Neighbor types

Each lattice defines up to three shells of neighbors:

| Shell | Name | Square | Triangular | Honeycomb | Cubic | Chain |
|---|---|---|---|---|---|---|
| 1st | **NN** (J1) | 4 | 6 | 3 | 6 | 2 |
| 2nd | **NNN** (J2) | 4 | 6 | 6 | 12 | 2 |
| 3rd | **TNN** (J3) | 4 | 6 | 3 | 8 | 2 |

Use J2 and J3 couplings to activate further neighbor interactions. See [Frustrated Magnetism](frustrated-magnetism.md) for details.

## What's next?

- **[Frustrated Magnetism](frustrated-magnetism.md)** — competing J1-J2 interactions on these lattices
- **[Cluster Algorithms](cluster-algorithms.md)** — Wolff and Swendsen-Wang on any lattice
