"""Writer for legacy (schema v1) HDF5 fixture files.

Reproduces exactly what ``_write_metadata`` wrote in mcising <= 0.23.0
(``python/mcising/io.py@cf5d068:496-502``): a ``metadata`` group with only
``version``, ``config_json``, and optionally ``elapsed_seconds`` — no
``schema_version`` attribute. Temperature groups were (and are) written
by ``_write_temperature_group``, whose layout is unchanged between
schema 1 and 2, so the current writer is reused for them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_legacy_hdf5(
    path: Path,
    *,
    version: str | None = "0.2.0",
    config_json: str | None = "{}",
    temperatures: tuple[float, ...] = (3.0, 2.0),
    n_samples: int = 5,
    size: int = 4,
    seed: int = 7,
) -> None:
    """Write an old-schema results file with synthetic per-T arrays.

    Parameters
    ----------
    path : Path
        Output file path.
    version : str or None
        Value for the ``version`` attribute; None omits the attribute
        (some very old files predate it).
    config_json : str or None
        Value for the ``config_json`` attribute; None omits it.
    temperatures : tuple[float, ...]
        Temperature groups to create.
    n_samples : int
        Measurements per temperature.
    size : int
        Linear lattice size of the synthetic configurations.
    seed : int
        Seed for the synthetic data arrays.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        if version is not None:
            meta.attrs["version"] = version
        if config_json is not None:
            meta.attrs["config_json"] = config_json
        for temp in temperatures:
            grp = f.create_group(f"T={temp:.6f}")
            grp.attrs["temperature"] = temp
            grp.create_dataset("energy", data=rng.normal(-1.0, 0.1, n_samples))
            grp.create_dataset("magnetization", data=rng.normal(0.0, 0.5, n_samples))
            spins = rng.choice(
                np.array([-1, 1], dtype=np.int8), size=(n_samples, size, size)
            )
            grp.create_dataset("configurations", data=spins)
