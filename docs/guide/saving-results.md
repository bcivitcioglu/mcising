# Saving Results

mcising provides three ways to persist simulation data: HDF5 (full data), JSON (summaries), and checkpointing (crash recovery).

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
│   ├── version
│   └── config_json
├── T=2.269/
│   ├── energy           (n_samples,)
│   ├── magnetization    (n_samples,)
│   └── configurations   (n_samples, L, L)
└── T=1.500/
    └── ...
```

## JSON — lightweight summary

```python
from mcising import save_json_summary

# Save means and standard deviations only (no large arrays)
save_json_summary(results, "summary.json")
```

Good for quick inspection, logging, or feeding into other tools.

## Checkpointing — crash recovery

For long simulations, checkpoint after each temperature so you don't lose progress:

```python
from mcising import checkpoint_run

checkpoint_run(
    config,
    checkpoint_path="checkpoint.h5",
    output_path="results.h5",
)
```

If interrupted, resume from where you left off:

```python
checkpoint_run(
    config,
    checkpoint_path="checkpoint.h5",
    output_path="results.h5",
    resume=True,
)
```

Or from the CLI:

```bash
mcising run -L 32 --checkpoint sim.h5
mcising run -L 32 --checkpoint sim.h5 --resume
```

Each completed temperature is flushed to disk immediately — a crash only loses the current temperature's data.
