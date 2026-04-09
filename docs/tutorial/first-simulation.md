# Your First Simulation

Let's run a Monte Carlo simulation of the 2D Ising model. By the end of this page, you'll have simulated a phase transition and plotted the results.

## Install mcising

=== "uv"

    ```bash
    uv add mcising
    ```

=== "pip"

    ```bash
    pip install mcising
    ```

## Create a simulation

Every simulation starts with a configuration. Let's set up a 32x32 square lattice and scan through three temperatures:

```python
from mcising import Simulation, SimulationConfig, LatticeConfig

config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    n_thermalization=100,
    seed=42,
)
```

- **`size=32`** — a 32x32 lattice (1,024 spins)
- **`j1=1.0`** — ferromagnetic nearest-neighbor coupling
- **`temperatures`** — we scan three: above Tc, at Tc (2.269), and below Tc
- **`n_sweeps=1000`** — 1,000 measurement sweeps per temperature
- **`seed=42`** — deterministic results (same seed = same output, always)

## Run it

```python
sim = Simulation(config)
results = sim.run()
```

You'll see a progress bar as mcising sweeps through the temperatures from high to low (cool-down approach).

## Look at the results

```python
for T in results.temperatures:
    E = results.energy[T].mean()
    M = abs(results.magnetization[T]).mean()
    print(f"T={T:.3f}: <E>={E:.4f}, <|M|>={M:.4f}")
```

You should see something like:

```
T=3.000: <E>=-0.8523, <|M|>=0.2841
T=2.269: <E>=-1.4531, <|M|>=0.6712
T=1.500: <E>=-1.9375, <|M|>=0.9843
```

The magnetization jumps from ~0.28 (disordered) to ~0.98 (ordered) as you cross the critical temperature. That's the Ising phase transition.

!!! info "Why these numbers?"
    At T=3.0 (above Tc), thermal fluctuations dominate — spins point randomly, so |M| is small. At T=1.5 (below Tc), the exchange coupling wins — nearly all spins align, so |M| approaches 1.0. The critical temperature Tc=2.269 is the exact transition point for the 2D square lattice.

## Save your results

```python
from mcising import save_hdf5, save_json_summary

# Full data (configurations, time series)
save_hdf5(results, "results.h5")

# Lightweight summary (means and standard deviations)
save_json_summary(results, "summary.json")
```

## Plot the phase transition

```python
from mcising import plot_observables

fig = plot_observables(results)
fig.savefig("phase_transition.png")
```

This produces a plot of energy and magnetization vs temperature with error bars.

## Use the CLI instead

You can do the same thing from the command line:

```bash
mcising run -L 32 --seed 42 -T 3.0 -T 2.269 -T 1.5 -o results.h5
```

## What's next?

- **[Lattice Types](lattice-types.md)** — try triangular, honeycomb, cubic, or chain lattices
- **[Frustrated Magnetism](frustrated-magnetism.md)** — add J2 and J3 couplings for competing interactions
- **[Parallel Execution](parallel-execution.md)** — use all your CPU cores with `mode=ExecutionMode.INDEPENDENT`
