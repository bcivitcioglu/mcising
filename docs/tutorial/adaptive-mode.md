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

That's it. For each temperature, mcising will automatically:

1. **Anneal** with a cool-down ramp (never analyzed — its energy trace
   is non-stationary by construction)
2. **Probe** with a fixed-temperature diagnostic series and **detect
   thermalization** on it using MSER (Marginal Standard Error Rule),
   extending the run while stationarity is not detected (up to
   `max_thermalization_sweeps`; a `UserWarning` is raised if the budget
   runs out first)
3. **Estimate the autocorrelation time** with Sokal's windowing method
   on the stationary tail of that fixed-temperature series
4. **Set measurement spacing** to `2 * tau_int` for approximately independent samples
5. **Collect** at least `min_independent_samples` measurements (with a
   `UserWarning` if the `max_total_sweeps` budget cannot afford them)

## Inspect diagnostics

After the run, check what adaptive mode decided:

```python
for T in results.temperatures:
    diag = results.adaptive_diagnostics[T]
    print(
        f"T={T:.3f}: "
        f"tau_int={diag.tau_int:.1f}, "
        f"interval={diag.measurement_interval}, "
        f"samples={diag.n_samples}, "
        f"stationary_sweeps={diag.stationary_sweeps}"
    )
```

You'll see that `tau_int` is larger near Tc (critical slowing down) and the measurement interval adjusts accordingly. `stationary_sweeps` records how many fixed-temperature sweeps the estimates are based on — the annealing ramp is never part of them.

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

**Ramp / diagnostics split:** The cool-down ramp is pure annealing. All statistical decisions come from a fixed-temperature energy series recorded *after* the ramp — estimating `tau_int` across a temperature ramp would measure the ramp, not the physics.

**MSER (thermalization detection):** Scans the fixed-temperature energy series to find the truncation point that minimizes the marginal standard error, evaluating every candidate in the first half exactly. Points before the truncation point are discarded as transient; an argmin at or beyond the midpoint means the series cannot demonstrate stationarity and is reported as not thermalized.

**Sokal windowing (autocorrelation):** Computes the integrated autocorrelation time `tau_int` from the stationary tail of the fixed-temperature series. Uses a self-consistent cutoff to avoid summing noise.

**Measurement interval:** Set to `tau_multiplier * tau_int` (default 2.0). With `tau_multiplier=2`, consecutive samples are approximately 86% independent.

Both algorithms run in O(N) time and add negligible overhead to the simulation.
