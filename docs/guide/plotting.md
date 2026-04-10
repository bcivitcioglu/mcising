# Plotting

mcising includes plotting functions for all key observables, lattice visualization, and a bulk export tool. Every plot function accepts either a `SimulationResults` object or an HDF5 file path.

## Thermodynamic quantities

Each function plots one quantity vs temperature, with error bars computed from the measurement samples.

```python
from mcising import (
    Simulation, SimulationConfig, LatticeConfig, save_hdf5,
    plot_energy, plot_magnetization, plot_specific_heat, plot_susceptibility,
)
import numpy as np

config = SimulationConfig(
    lattice=LatticeConfig(size=32),
    temperatures=tuple(np.linspace(1.5, 3.5, 20)),
    n_sweeps=1000,
    measurement_interval=10,
)
results = Simulation(config).run()
save_hdf5(results, "results.h5")

# Plot from memory or from file — both work
plot_energy(results).savefig("energy.png")
plot_magnetization("results.h5").savefig("magnetization.png")
plot_specific_heat("results.h5").savefig("specific_heat.png")
plot_susceptibility("results.h5").savefig("susceptibility.png")
```

### Comparing different coupling configurations

Pass a list of file paths to overlay results with auto-generated legends:

```python
plot_energy(["j2_0.0.h5", "j2_0.3.h5", "j2_0.5.h5"])
```

Each curve is labeled with the coupling parameters extracted from the HDF5 metadata.

## Spin configurations

### Single temperature — all configs or one

```python
from mcising import plot_lattice

# Show ALL configurations at T=2.269 side by side
plot_lattice("results.h5", temperature=2.269)

# Show just configuration #3
plot_lattice("results.h5", temperature=2.269, n=3)
```

If you request an invalid index, you get a descriptive error:

```
ValueError: Config index 999 out of range. Temperature T=2.269
has 50 configurations (n=0..49).
```

### Bulk export to zip

Export every lattice configuration as a PNG in a zip file:

```python
from mcising import export_lattices

# Tree mode: folders per temperature
export_lattices("results.h5", "lattices.zip")
# → square_32x32_J1=1.0_metropolis/T=2.2690/config_001.png

# Flat mode: all PNGs in one folder
export_lattices("results.h5", "lattices.zip", flat=True)
# → square_32x32_J1=1.0_metropolis_T=2.2690_config_001.png

# Export only specific temperatures
export_lattices("results.h5", "lattices.zip", temperatures=[2.269, 1.5])
```

Filenames encode lattice type, size, couplings, algorithm, temperature, and sample number.

## Diagnostic plots

### Energy time series

Check if thermalization was sufficient — the trace should be stationary (no drift):

```python
from mcising import plot_energy_timeseries

plot_energy_timeseries("results.h5", temperature=2.269)
```

### Magnetization histogram

See the distribution of magnetization at a given temperature. Bimodal below Tc, Gaussian above:

```python
from mcising import plot_magnetization_histogram

plot_magnetization_histogram("results.h5", temperature=2.269)
```

### Correlation function

Requires `compute_correlation=True` in the simulation config:

```python
from mcising import plot_correlation

plot_correlation("results.h5", temperature=2.269)
```

## Summary table

No plotting needed — just print a Rich table:

```python
results.summary()
```

```
                    Simulation Results
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃      T ┃   <E>/N ┃ <|M|>/N ┃   Cv/N ┃   chi/N ┃ samples ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ 1.5000 │ -1.9484 │  0.9859 │ 0.1992 │  0.0246 │      50 │
│ 2.2690 │ -1.4531 │  0.6712 │ 1.8420 │ 12.5430 │      50 │
│ 3.5000 │ -0.6630 │  0.0605 │ 0.2657 │  1.5562 │      50 │
└────────┴─────────┴─────────┴────────┴─────────┴─────────┘
```

See the [API Reference](../reference/plotting.md) for full function signatures.
