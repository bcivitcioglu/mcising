# Plotting

mcising includes three visualization functions built on matplotlib.

## Spin configuration

```python
from mcising import Simulation, SimulationConfig, LatticeConfig, plot_lattice

sim = Simulation(SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(2.269,),
    n_sweeps=100,
))
results = sim.run(show_progress=False)

fig = plot_lattice(results.configurations[2.269][-1], title="T=2.269")
fig.savefig("spins.png")
```

Shows a heatmap of the spin configuration: blue = +1, red = -1.

## Observables vs temperature

```python
from mcising import plot_observables

fig = plot_observables(results)
fig.savefig("observables.png")
```

Plots energy and magnetization vs temperature with error bars. Pass `quantities=("energy",)` or `quantities=("magnetization",)` to show only one.

## Correlation function

```python
from mcising import plot_correlation

config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=(2.269,),
    n_sweeps=100,
    compute_correlation=True,
)
results = Simulation(config).run(show_progress=False)

fig = plot_correlation(results, temperature=2.269)
fig.savefig("correlation.png")
```

Shows C(r) vs distance. Requires `compute_correlation=True` in the config.

See the [API Reference](../reference/plotting.md) for full function signatures.
