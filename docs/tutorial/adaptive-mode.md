# Adaptive Mode

How many thermalization sweeps do you need? How often should you measure? Adaptive mode answers both questions automatically.

## The problem

With fixed parameters, you're guessing:

- Too few thermalization sweeps → measurements are biased by the initial state
- Too many → wasting computation time
- Measurement interval too small → correlated samples (not independent)
- Measurement interval too large → wasting sweeps between measurements

Near the critical temperature, autocorrelation times can be 10-100x longer than at high temperature. A single fixed interval doesn't work well across a temperature scan.

## Enable adaptive mode

```python
from mcising import (
    Simulation, SimulationConfig, LatticeConfig, AdaptiveConfig,
)

config = SimulationConfig(
    lattice=LatticeConfig(size=64),
    temperatures=(3.0, 2.269, 1.5),
    adaptive=AdaptiveConfig(
        enabled=True,
        min_independent_samples=200,
    ),
    seed=42,
)

results = Simulation(config).run()
```

That's it. mcising will automatically:

1. **Detect thermalization** using MSER (Marginal Standard Error Rule)
2. **Estimate autocorrelation time** using Sokal's windowing method
3. **Set measurement spacing** to `2 * tau_int` for approximately independent samples
4. **Collect** at least `min_independent_samples` measurements

## Inspect diagnostics

After the run, check what adaptive mode decided:

```python
for T in results.temperatures:
    diag = results.adaptive_diagnostics[T]
    print(
        f"T={T:.3f}: "
        f"tau_int={diag.tau_int:.1f}, "
        f"interval={diag.measurement_interval}, "
        f"samples={diag.n_samples}"
    )
```

You'll see that `tau_int` is larger near Tc (critical slowing down) and the measurement interval adjusts accordingly.

## Configuration options

```python
AdaptiveConfig(
    enabled=True,
    min_thermalization_sweeps=200,       # minimum warmup
    max_thermalization_sweeps=10_000,    # cap to prevent runaway
    c_window=6.0,                        # Sokal windowing constant
    min_independent_samples=100,         # target sample count
    max_total_sweeps=100_000,            # hard budget cap
    tau_multiplier=2.0,                  # interval = tau_multiplier * tau_int
)
```

!!! tip "When to use adaptive mode"
    Use it when scanning a wide temperature range on large lattices. Near Tc, autocorrelation times diverge — adaptive mode handles this gracefully. For quick exploratory runs on small lattices, fixed parameters are fine.

## How it works

**MSER (thermalization detection):** Scans the energy time series to find the truncation point that minimizes the marginal standard error. Points before this are discarded as transient.

**Sokal windowing (autocorrelation):** Computes the integrated autocorrelation time `tau_int` from the stationary tail of the energy series. Uses a self-consistent cutoff to avoid summing noise.

**Measurement interval:** Set to `tau_multiplier * tau_int` (default 2.0). With `tau_multiplier=2`, consecutive samples are approximately 86% independent.

Both algorithms run in O(N) time and add negligible overhead to the simulation.
