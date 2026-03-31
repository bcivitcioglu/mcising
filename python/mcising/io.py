"""HDF5 and JSON I/O for simulation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from mcising.simulation import Simulation, SimulationResults

__all__: Final[list[str]] = [
    "save_hdf5",
    "load_hdf5",
    "save_json_summary",
    "init_checkpoint_file",
    "save_temperature_group",
    "load_completed_temperatures",
    "checkpoint_run",
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
        _write_metadata(f, results)
        for temp in results.temperatures:
            _write_temperature_group(f, temp, results)


def init_checkpoint_file(
    path: str | Path, results: SimulationResults
) -> None:
    """Create an HDF5 checkpoint file with only the metadata group.

    Parameters
    ----------
    path : str or Path
        Output file path.
    results : SimulationResults
        Results object (used for metadata extraction).
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        _write_metadata(f, results)


def save_temperature_group(
    path: str | Path, temperature: float, results: SimulationResults
) -> None:
    """Append a single temperature group to an existing HDF5 checkpoint file.

    Opens the file in append mode, writes the data for one temperature,
    and closes. Safe against crashes (file is never left open between
    temperature points).

    Parameters
    ----------
    path : str or Path
        HDF5 file path (must already exist with metadata).
    temperature : float
        The temperature to write.
    results : SimulationResults
        Results object containing data for this temperature.
    """
    import h5py

    with h5py.File(path, "a") as f:
        _write_temperature_group(f, temperature, results)


def load_completed_temperatures(path: str | Path) -> set[float]:
    """Read an HDF5 checkpoint and return the set of completed temperatures.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.

    Returns
    -------
    set[float]
        Temperatures that have already been saved.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        temps: set[float] = set()
        for key in f.keys():
            if key.startswith("T="):
                temps.add(float(f[key].attrs["temperature"]))
        return temps


def checkpoint_run(
    sim: Simulation,
    path: str | Path,
    *,
    show_progress: bool = True,
    resume: bool = False,
    checkpoint_interval: int = 1,
) -> SimulationResults:
    """Run a simulation with periodic HDF5 checkpointing.

    After every ``checkpoint_interval`` temperatures complete, results
    are appended to the checkpoint file. If the process is interrupted,
    already-completed temperatures are preserved.

    Parameters
    ----------
    sim : Simulation
        Configured simulation instance.
    path : str or Path
        HDF5 checkpoint file path.
    show_progress : bool
        Whether to display progress bars.
    resume : bool
        If True and the file exists, skip already-completed temperatures.
    checkpoint_interval : int
        Save checkpoint every N completed temperatures. Default is 1
        (save after every temperature). Use higher values for speed
        at the cost of less frequent saves.

    Returns
    -------
    SimulationResults
        Complete simulation results (including resumed data).
    """
    path = Path(path)
    skip_temps: frozenset[float] = frozenset()
    resumed_results: SimulationResults | None = None

    if resume and path.exists():
        skip_temps = frozenset(load_completed_temperatures(path))
        if skip_temps:
            resumed_results = load_hdf5(path)
            _restore_simulation_state(path, sim)

    temp_counter = 0
    unsaved_temps: list[float] = []

    def _on_complete(temperature: float, results: SimulationResults) -> None:
        nonlocal temp_counter
        temp_counter += 1
        unsaved_temps.append(temperature)

        if temp_counter % checkpoint_interval == 0:
            if not path.exists():
                init_checkpoint_file(path, results)
            for t in unsaved_temps:
                save_temperature_group(path, t, results)
            _save_simulation_state(path, sim)
            unsaved_temps.clear()

    results = sim.run(
        show_progress=show_progress,
        on_temperature_complete=_on_complete,
        skip_temperatures=skip_temps if skip_temps else None,
    )

    # Flush any remaining unsaved temperatures
    if unsaved_temps:
        if not path.exists():
            init_checkpoint_file(path, results)
        for t in unsaved_temps:
            save_temperature_group(path, t, results)
        _save_simulation_state(path, sim)
        unsaved_temps.clear()

    # Merge resumed data into results
    if resumed_results is not None:
        for temp in skip_temps:
            if temp not in results.temperatures:
                results.temperatures.append(temp)
            if temp in resumed_results.energy:
                results.energy[temp] = resumed_results.energy[temp]
            if temp in resumed_results.magnetization:
                results.magnetization[temp] = resumed_results.magnetization[temp]
            if temp in resumed_results.configurations:
                results.configurations[temp] = resumed_results.configurations[temp]
            if (
                resumed_results.correlation_function is not None
                and temp in resumed_results.correlation_function
            ):
                if results.correlation_function is None:
                    results.correlation_function = {}
                results.correlation_function[temp] = (
                    resumed_results.correlation_function[temp]
                )
            if (
                resumed_results.correlation_length is not None
                and temp in resumed_results.correlation_length
            ):
                if results.correlation_length is None:
                    results.correlation_length = {}
                results.correlation_length[temp] = (
                    resumed_results.correlation_length[temp]
                )
        # Re-sort temperatures descending
        results.temperatures.sort(reverse=True)

    # Update elapsed_seconds in the checkpoint file
    import h5py

    if path.exists() and "elapsed_seconds" in results.metadata:
        elapsed = float(cast(float, results.metadata["elapsed_seconds"]))
        if resumed_results is not None and "elapsed_seconds" in (
            resumed_results.metadata
        ):
            elapsed += float(
                cast(float, resumed_results.metadata["elapsed_seconds"])
            )
            results.metadata["elapsed_seconds"] = elapsed
        with h5py.File(path, "a") as f:
            if "metadata" in f:
                f["metadata"].attrs["elapsed_seconds"] = elapsed

    return results


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


def _save_simulation_state(path: str | Path, sim: Simulation) -> None:
    """Save the simulation's spin and RNG state to the checkpoint file."""
    import h5py

    with h5py.File(path, "a") as f:
        if "sim_state" in f:
            del f["sim_state"]
        state = f.create_group("sim_state")
        state.create_dataset("spins", data=sim.spins)
        rng_bytes = bytes(sim._core.get_rng_state())
        state.create_dataset(
            "rng_state", data=np.frombuffer(rng_bytes, dtype=np.uint8)
        )


def _restore_simulation_state(path: str | Path, sim: Simulation) -> None:
    """Restore the simulation's spin and RNG state from a checkpoint file."""
    import h5py

    with h5py.File(path, "r") as f:
        if "sim_state" not in f:
            return
        state = f["sim_state"]
        if "spins" in state:
            sim.spins = np.array(state["spins"])
        if "rng_state" in state:
            rng_array = np.array(state["rng_state"], dtype=np.uint8)
            sim._core.set_rng_state(list(rng_array))


def _write_metadata(f: Any, results: SimulationResults) -> None:
    """Write the metadata group to an HDF5 file handle."""
    meta = f.create_group("metadata")
    meta.attrs["version"] = str(results.metadata.get("version", "unknown"))
    meta.attrs["config_json"] = _config_to_json(results.metadata.get("config"))
    if "elapsed_seconds" in results.metadata:
        meta.attrs["elapsed_seconds"] = results.metadata["elapsed_seconds"]


def _write_temperature_group(
    f: Any, temp: float, results: SimulationResults
) -> None:
    """Write a single temperature group to an HDF5 file handle."""
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
