"""HDF5 and JSON I/O for simulation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from mcising.simulation import SimulationResults

__all__: Final[list[str]] = [
    "save_hdf5",
    "load_hdf5",
    "save_json_summary",
]


def save_hdf5(results: SimulationResults, path: str | Path) -> None:
    """Save simulation results to an HDF5 file.

    File structure::

        results.h5
        ├── metadata/
        │   ├── version       (attribute)
        │   └── config_json   (attribute)
        ├── T=2.269/
        │   ├── configurations  (n_samples x L x L, int8)
        │   ├── energy          (n_samples, float64)
        │   ├── magnetization   (n_samples, float64)
        │   ├── correlation_function  (n_distances, float64) [optional]
        │   ├── correlation_distances (n_distances, float64) [optional]
        │   └── correlation_length    (n_samples, float64)  [optional]
        └── ...

    Parameters
    ----------
    results : SimulationResults
        The simulation results to save.
    path : str or Path
        Output file path (should end in .h5 or .hdf5).
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["version"] = str(results.metadata.get("version", "unknown"))
        meta.attrs["config_json"] = _config_to_json(results.metadata.get("config"))
        if "elapsed_seconds" in results.metadata:
            meta.attrs["elapsed_seconds"] = results.metadata["elapsed_seconds"]

        # Per-temperature data
        for temp in results.temperatures:
            group_name = f"T={temp:.6f}"
            grp = f.create_group(group_name)
            grp.attrs["temperature"] = temp

            if temp in results.energy:
                grp.create_dataset("energy", data=results.energy[temp])
            if temp in results.magnetization:
                grp.create_dataset("magnetization", data=results.magnetization[temp])
            if temp in results.configurations:
                grp.create_dataset(
                    "configurations",
                    data=results.configurations[temp],
                    compression="gzip",
                    compression_opts=4,
                )

            if (
                results.correlation_function is not None
                and temp in results.correlation_function
            ):
                distances, correlations = results.correlation_function[temp]
                grp.create_dataset("correlation_distances", data=distances)
                grp.create_dataset("correlation_function", data=correlations)

            if (
                results.correlation_length is not None
                and temp in results.correlation_length
            ):
                grp.create_dataset(
                    "correlation_length", data=results.correlation_length[temp]
                )


def load_hdf5(path: str | Path) -> SimulationResults:
    """Load simulation results from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    SimulationResults
        The loaded simulation results.
    """
    import h5py

    path = Path(path)

    with h5py.File(path, "r") as f:
        # Metadata
        metadata: dict[str, object] = {}
        if "metadata" in f:
            meta = f["metadata"]
            metadata["version"] = str(meta.attrs.get("version", "unknown"))
            if "elapsed_seconds" in meta.attrs:
                metadata["elapsed_seconds"] = float(meta.attrs["elapsed_seconds"])

        # Discover temperature groups
        temp_groups = [
            key
            for key in f.keys()
            if key.startswith("T=")  # noqa: SIM118
        ]
        temperatures: list[float] = []
        energy: dict[float, np.ndarray[Any, Any]] = {}
        magnetization: dict[float, np.ndarray[Any, Any]] = {}
        configurations: dict[float, np.ndarray[Any, Any]] = {}
        correlation_function: dict[
            float, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = {}
        correlation_length: dict[float, np.ndarray[Any, Any]] = {}

        for group_name in sorted(temp_groups):
            grp = f[group_name]
            temp = float(grp.attrs["temperature"])
            temperatures.append(temp)

            if "energy" in grp:
                energy[temp] = np.array(grp["energy"])
            if "magnetization" in grp:
                magnetization[temp] = np.array(grp["magnetization"])
            if "configurations" in grp:
                configurations[temp] = np.array(grp["configurations"])
            if "correlation_distances" in grp and "correlation_function" in grp:
                correlation_function[temp] = (
                    np.array(grp["correlation_distances"]),
                    np.array(grp["correlation_function"]),
                )
            if "correlation_length" in grp:
                correlation_length[temp] = np.array(grp["correlation_length"])

        return SimulationResults(
            temperatures=temperatures,
            energy=energy,
            magnetization=magnetization,
            configurations=configurations,
            correlation_function=correlation_function if correlation_function else None,
            correlation_length=correlation_length if correlation_length else None,
            metadata=metadata,
        )


def save_json_summary(results: SimulationResults, path: str | Path) -> None:
    """Save a JSON summary of simulation results (no large arrays).

    Parameters
    ----------
    results : SimulationResults
        The simulation results to summarize.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "version": results.metadata.get("version", "unknown"),
        "temperatures": results.temperatures,
        "results": {},
    }

    results_dict: dict[str, object] = {}
    for temp in results.temperatures:
        entry: dict[str, float] = {}
        if temp in results.energy:
            entry["mean_energy"] = float(np.mean(results.energy[temp]))
            entry["std_energy"] = float(np.std(results.energy[temp]))
        if temp in results.magnetization:
            entry["mean_abs_magnetization"] = float(
                np.mean(np.abs(results.magnetization[temp]))
            )
            entry["std_magnetization"] = float(np.std(results.magnetization[temp]))
        if (
            results.correlation_length is not None
            and temp in results.correlation_length
        ):
            entry["mean_correlation_length"] = float(
                np.mean(results.correlation_length[temp])
            )
        results_dict[f"{temp:.6f}"] = entry

    summary["results"] = results_dict

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _config_to_json(config: Any) -> str:
    """Convert a config object to a JSON string for HDF5 metadata."""
    if config is None:
        return "{}"
    try:
        from dataclasses import asdict

        d = asdict(config)
        return json.dumps(d, default=str, indent=2)
    except (TypeError, ValueError):
        return str(config)
