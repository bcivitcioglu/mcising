# Saving Results

mcising provides three ways to persist simulation data: HDF5 (full data), JSON (summaries), and checkpointing (crash recovery).

The examples below persist this run:

```python
from mcising import LatticeConfig, Simulation, SimulationConfig

config = SimulationConfig(
    lattice=LatticeConfig(size=16),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=500,
)
results = Simulation(config).run()
```

## HDF5 — full data

```python
from mcising import save_hdf5, load_hdf5

# Save everything: energy time series, magnetization, spin configurations
save_hdf5(results, "results.h5")

# Load it back
loaded = load_hdf5("results.h5")
print(loaded.energy[2.269].mean())
```

HDF5 files are structured by temperature:

```
results.h5
├── metadata/
│   ├── schema_version   (3)
│   ├── version          (mcising version that wrote the file)
│   ├── config_json      (full config as JSON)
│   ├── seed
│   ├── mode
│   ├── algorithm
│   ├── git_commit       (when built from a git checkout)
│   └── elapsed_seconds
├── T=2.269/
│   ├── energy           (n_samples,)
│   ├── magnetization    (n_samples,)
│   ├── configurations   (n_samples, *lattice shape) — see below
│   └── statistics/      (derived estimates as attributes: n_samples,
│                         tau_int, and value + *_error pairs for energy,
│                         magnetization, abs_magnetization, specific_heat,
│                         susceptibility, binder_cumulant)
└── T=1.500/
    └── ...
```

`configurations` has the lattice's own shape after the sample axis:
`(L, L)` for the square and triangular lattices, `(L, L, 2)` for the
honeycomb (two sites per cell), `(L, L, L)` for the cubic lattice and
`(L,)` for the chain — the same array `IsingSimulation.get_spins()`
returns.

The `statistics` subgroup (schema 3) exists for external tools —
`h5dump`, pandas, a referee's notebook — so a saved file quotes its
observables with uncertainties without needing mcising at all. Loading
**ignores** it: `load_hdf5` restores the raw series and
`results.statistics(T)` recomputes everything, so the derived numbers
can never drift out of sync with the data. Values that cannot be
estimated (for example jackknife errors of a 2-sample series) are
omitted rather than stored as NaN.

### Provenance

Every file records the code and run that produced it: the writing
mcising version, the metadata schema version, the seed, execution mode,
algorithm, the full configuration, and (for development builds) the git
commit. `load_hdf5` restores all of it — including the
`SimulationConfig` object under `results.metadata["config"]`, so plot
legends, export prefixes, and per-site observables work on loaded files
exactly as on in-memory results. `save_json_summary` carries the same
fields.

Files written by mcising ≤ 0.23.0 (no `schema_version` attribute) still
load; their config is reconstructed from `config_json` when possible,
and a file that predates the version stamp reports version `"unknown"`.
Files written by a *newer* schema than your mcising supports are
refused with a clear error instead of loading incompletely. Resuming a
pre-0.24 checkpoint keeps its original metadata: a file records the
code that created it.

## JSON — lightweight summary

```python
from mcising import save_json_summary

# Save per-temperature estimates with standard errors (no large arrays)
save_json_summary(results, "summary.json")
```

Each temperature entry carries the means with their standard errors
(`energy_error`, `abs_magnetization_error`), the derived quantities
with jackknife errors (`specific_heat`, `susceptibility`,
`binder_cumulant` and their `*_error` twins), `tau_int`, and
`n_samples`. Unquotable values are omitted — never written as null or
NaN — so the output is always strict JSON.

Good for quick inspection, logging, or feeding into other tools.

## Checkpointing — crash recovery

For long simulations, checkpoint completed temperatures so you don't lose progress:

```python
from mcising import Simulation, checkpoint_run

sim = Simulation(config)
results = checkpoint_run(sim, "checkpoint.h5")
```

If interrupted, resume from where you left off (the config must match the
one the checkpoint was written with — only `temperatures` may differ, so a
scan can be extended):

```python
results = checkpoint_run(Simulation(config), "checkpoint.h5", resume=True)
```

Without `resume=True`, an existing checkpoint file is an error — a new
run never silently mixes its data into a file from an earlier one.
Delete the file or pick a new path to start fresh.

Or from the CLI:

```bash
mcising run -L 32 --checkpoint sim.h5
mcising run -L 32 --checkpoint sim.h5 --resume
```

Checkpoint granularity depends on the execution mode:

- **Cooldown**: each completed temperature is flushed to disk immediately —
  a crash only loses the current temperature's data.
- **Independent**: the batch runs in parallel and every temperature is
  saved when it returns; on resume, only the missing temperatures run, and
  they keep the RNG streams they would have had in an uninterrupted run.
- **Parallel tempering**: all-or-nothing — the replicas form one coupled
  ensemble, so a partially completed ladder cannot be resumed.
