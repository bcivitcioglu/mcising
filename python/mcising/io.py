"""HDF5 and JSON I/O for simulation results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from mcising._provenance import HDF5_SCHEMA_VERSION, git_commit, package_version
from mcising.config import ExecutionMode, SimulationConfig
from mcising.exceptions import ConfigurationError
from mcising.simulation import AdaptiveDiagnostics, Simulation, SimulationResults
from mcising.statistics import ObservableStatistics

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

    File structure (metadata schema v3; files without ``schema_version``
    were written by mcising <= 0.23.0 and load through a legacy path;
    schema 2 files lack the ``statistics`` subgroup)::

        results.h5
        ├── metadata/
        │   ├── schema_version  (attribute, int)
        │   ├── version         (attribute, mcising version that wrote the file)
        │   ├── config_json     (attribute, full config as JSON)
        │   ├── seed            (attribute) [when a config is recorded]
        │   ├── mode            (attribute) [when a config is recorded]
        │   ├── algorithm       (attribute) [when a config is recorded]
        │   ├── git_commit      (attribute) [when built from a git checkout]
        │   └── elapsed_seconds (attribute) [when known]
        ├── T=2.269/
        │   ├── configurations  (n_samples x L x L, int8)
        │   ├── energy          (n_samples, float64)
        │   ├── magnetization   (n_samples, float64)
        │   ├── correlation_function  (n_distances, float64) [optional]
        │   ├── correlation_distances (n_distances, float64) [optional]
        │   ├── correlation_length    (n_samples, float64)  [optional]
        │   └── statistics/     (derived observable estimates, attributes:
        │       n_samples, tau_int, and value + ``*_error`` pairs for
        │       energy, magnetization, abs_magnetization, specific_heat,
        │       susceptibility, binder_cumulant; non-finite values are
        │       omitted. Written for external inspection only — loading
        │       recomputes statistics from the raw series.)
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


def init_checkpoint_file(path: str | Path, results: SimulationResults) -> None:
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

    Checkpoint granularity depends on the execution mode: cooldown
    saves after each temperature; independent mode computes the batch
    in parallel and saves every temperature when it returns; parallel
    tempering is all-or-nothing (the replicas form one coupled
    ensemble, so a partial ladder cannot be resumed).

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
        The stored config must match ``sim.config`` (temperatures may
        differ, so a scan can be extended on resume). Resuming a file
        written by mcising <= 0.23.0 keeps its original metadata schema:
        a file records the code that created it.
    checkpoint_interval : int
        Save checkpoint every N completed temperatures. Default is 1
        (save after every temperature). Use higher values for speed
        at the cost of less frequent saves.

    Returns
    -------
    SimulationResults
        Complete simulation results (including resumed data).

    Raises
    ------
    ConfigurationError
        When ``resume=False`` but ``path`` already exists; on resume,
        when the checkpoint was written with a different config than
        ``sim.config`` (or its config record is unreadable), or when a
        parallel-tempering ladder is only partially complete.
    """
    path = Path(path)
    if not resume and path.exists():
        raise ConfigurationError(
            f"Checkpoint file {path} already exists. Pass resume=True "
            "(CLI: --resume) to continue it, delete the file, or choose "
            "another path; writing a new run into an existing file would "
            "mix two runs' data under one provenance record."
        )
    skip_temps: frozenset[float] = frozenset()
    resumed_results: SimulationResults | None = None
    restored = False

    if resume and path.exists():
        _check_resume_config(path, sim.config)
        skip_temps = frozenset(load_completed_temperatures(path))
        if skip_temps:
            resumed_results = load_hdf5(path)
            if sim.config.mode == ExecutionMode.COOLDOWN:
                # The Python-side sim only advances in cooldown mode; the
                # parallel modes build their own replicas in Rust, so
                # restoring spins/RNG here would imply a state continuity
                # those modes do not have.
                _restore_simulation_state(path, sim)
                restored = True

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
            if sim.config.mode == ExecutionMode.COOLDOWN:
                _save_simulation_state(path, sim)
            unsaved_temps.clear()

    # A restored core must survive run()'s default reset (P10 semantics).
    results = sim.run(
        reset=not restored,
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
        if sim.config.mode == ExecutionMode.COOLDOWN:
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
                results.correlation_length[temp] = resumed_results.correlation_length[
                    temp
                ]
        # Re-sort temperatures descending
        results.temperatures.sort(reverse=True)

    # Update elapsed_seconds in the checkpoint file
    import h5py

    if path.exists() and "elapsed_seconds" in results.metadata:
        elapsed = float(cast(float, results.metadata["elapsed_seconds"]))
        if resumed_results is not None and "elapsed_seconds" in (
            resumed_results.metadata
        ):
            elapsed += float(cast(float, resumed_results.metadata["elapsed_seconds"]))
            results.metadata["elapsed_seconds"] = elapsed
        with h5py.File(path, "a") as f:
            if "metadata" in f:
                f["metadata"].attrs["elapsed_seconds"] = elapsed

    return results


def load_hdf5(path: str | Path) -> SimulationResults:
    """Load simulation results from an HDF5 file.

    Restores the full provenance record (version, seed, mode, algorithm,
    and the ``SimulationConfig`` object) for schema v2 files; legacy files
    (mcising <= 0.23.0) load with a best-effort config reconstruction.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    SimulationResults
        The loaded simulation results.

    Raises
    ------
    ConfigurationError
        If the file's metadata schema is newer than this mcising supports.
    """
    import h5py

    path = Path(path)

    with h5py.File(path, "r") as f:
        metadata: dict[str, object] = {}
        if "metadata" in f:
            metadata = _read_metadata(f["metadata"], path)

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
        n_cluster_flips: dict[float, int] = {}
        adaptive_diagnostics: dict[float, AdaptiveDiagnostics] = {}

        for group_name in sorted(temp_groups):
            grp = f[group_name]
            temp = float(grp.attrs["temperature"])
            temperatures.append(temp)
            # Tolerant read: pre-P10 files carry no cluster-work attr.
            if "n_cluster_flips" in grp.attrs:
                n_cluster_flips[temp] = int(grp.attrs["n_cluster_flips"])

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
            if "adaptive_diagnostics" in grp:
                ad = grp["adaptive_diagnostics"]
                adaptive_diagnostics[temp] = AdaptiveDiagnostics(
                    thermalization_sweeps=int(ad.attrs["thermalization_sweeps"]),
                    truncation_point=int(ad.attrs["truncation_point"]),
                    is_thermalized=bool(ad.attrs["is_thermalized"]),
                    tau_int=float(ad.attrs["tau_int"]),
                    measurement_interval=int(ad.attrs["measurement_interval"]),
                    production_sweeps=int(ad.attrs["production_sweeps"]),
                    n_samples=int(ad.attrs["n_samples"]),
                    # Added in P09 (additive, no schema bump): files
                    # written before it load with the dataclass default.
                    stationary_sweeps=int(ad.attrs.get("stationary_sweeps", 0)),
                )

        return SimulationResults(
            temperatures=temperatures,
            energy=energy,
            magnetization=magnetization,
            configurations=configurations,
            correlation_function=correlation_function if correlation_function else None,
            correlation_length=correlation_length if correlation_length else None,
            n_cluster_flips=n_cluster_flips,
            adaptive_diagnostics=adaptive_diagnostics if adaptive_diagnostics else None,
            metadata=metadata,
        )


def save_json_summary(results: SimulationResults, path: str | Path) -> None:
    """Save a JSON summary of simulation results (no large arrays).

    Carries the same provenance fields as the HDF5 metadata group
    (version, schema_version, seed, mode, algorithm, git_commit, config);
    fields whose value is unknown are omitted rather than written as null.

    Parameters
    ----------
    results : SimulationResults
        The simulation results to summarize.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}
    for key in ("version", "schema_version", "seed", "mode", "algorithm", "git_commit"):
        if key in results.metadata:
            summary[key] = results.metadata[key]
    config = results.metadata.get("config")
    if config is not None:
        summary["config"] = json.loads(_config_to_json(config))
    if "elapsed_seconds" in results.metadata:
        summary["elapsed_seconds"] = results.metadata["elapsed_seconds"]
    summary["temperatures"] = results.temperatures
    summary["results"] = {}

    results_dict: dict[str, object] = {}
    for temp in results.temperatures:
        entry: dict[str, float | str] = {}

        def put(
            key: str, value: float, entry: dict[str, float | str] = entry
        ) -> None:
            # Non-finite values are omitted, never written as NaN
            # (invalid strict JSON) or null (P07 policy).
            if math.isfinite(value):
                entry[key] = value

        stats = results.statistics(temp)
        if temp in results.energy:
            entry["mean_energy"] = float(np.mean(results.energy[temp]))
            entry["std_energy"] = float(np.std(results.energy[temp]))
            put("energy_error", stats.energy.error)
        if temp in results.magnetization:
            entry["mean_abs_magnetization"] = float(
                np.mean(np.abs(results.magnetization[temp]))
            )
            entry["std_magnetization"] = float(np.std(results.magnetization[temp]))
            put("abs_magnetization_error", stats.abs_magnetization.error)
        if temp in results.energy or temp in results.magnetization:
            entry["n_samples"] = stats.n_samples
            put("tau_int", stats.tau_int)
            put("specific_heat", stats.specific_heat.value)
            put("specific_heat_error", stats.specific_heat.error)
            put("susceptibility", stats.susceptibility.value)
            put("susceptibility_error", stats.susceptibility.error)
            # Explicit convention label so the file is never ambiguous
            # about which chi it records (#39, P10).
            entry["susceptibility_kind"] = "connected"
            put("binder_cumulant", stats.binder_cumulant.value)
            put("binder_cumulant_error", stats.binder_cumulant.error)
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


def _load_stored_config(path: str | Path) -> dict[str, Any] | None:
    """Read the config dict recorded in a checkpoint's metadata group.

    Returns None when no config record exists (legacy files). Raises when
    a record exists but cannot be parsed back into a dict — an unreadable
    record must not silently disable the resume-mismatch check — or when
    the file's metadata schema is newer than this mcising supports.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "metadata" not in f:
            return None
        meta = f["metadata"]
        _schema_version_of(meta, Path(path))
        if "config_json" not in meta.attrs:
            return None
        raw = _as_str(meta.attrs["config_json"])

    try:
        stored = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        stored = None
    if not isinstance(stored, dict):
        raise ConfigurationError(
            f"Checkpoint {path} has an unreadable configuration record; "
            "cannot rule out a mismatched resume. Start a new checkpoint "
            "file, or load the data explicitly with load_hdf5()."
        )
    return stored


def _check_resume_config(path: str | Path, config: SimulationConfig) -> None:
    """Refuse to resume a checkpoint written with a different config.

    Compares the stored config dict against ``config`` serialized through
    the same writer (`_config_to_json`), over the keys both sides share.
    ``temperatures`` is exempt so a scan can be extended on resume; every
    other difference changes the ensemble or the array shapes being
    appended into one file.
    """
    stored = _load_stored_config(path)
    if stored is None:
        return

    current = json.loads(_config_to_json(config))
    if not isinstance(current, dict):
        raise ConfigurationError(
            "The current simulation config does not serialize to a "
            "comparable record; cannot rule out a mismatched resume."
        )

    shared_keys = sorted((set(stored) & set(current)) - {"temperatures"})
    mismatches = [
        f"{key}: checkpoint={stored[key]!r}, current={current[key]!r}"
        for key in shared_keys
        if stored[key] != current[key]
    ]
    if mismatches:
        raise ConfigurationError(
            "Checkpoint config does not match the current simulation "
            "config; resuming would mix incompatible ensembles in one "
            "file. Use the original config (temperatures may differ), or "
            "start a new checkpoint file. Mismatched fields: " + "; ".join(mismatches)
        )


def _save_simulation_state(path: str | Path, sim: Simulation) -> None:
    """Save the simulation's spin and RNG state to the checkpoint file."""
    import h5py

    with h5py.File(path, "a") as f:
        if "sim_state" in f:
            del f["sim_state"]
        state = f.create_group("sim_state")
        state.create_dataset("spins", data=sim.spins)
        rng_bytes = bytes(sim._core.get_rng_state())
        state.create_dataset("rng_state", data=np.frombuffer(rng_bytes, dtype=np.uint8))


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
    """Write the metadata group (schema v2) to an HDF5 file handle.

    Writer facts — ``schema_version``, ``version``, ``git_commit`` — are
    derived here rather than trusted from ``results.metadata``: they
    describe the code writing the file, so re-saving a loaded legacy file
    re-stamps the current version. Run facts — ``seed``, ``mode``,
    ``algorithm`` — come from the run's config and are omitted when
    unknown; absence is honest, a null would be a guess.
    """
    meta = f.create_group("metadata")
    meta.attrs["schema_version"] = HDF5_SCHEMA_VERSION
    meta.attrs["version"] = package_version()
    commit = git_commit()
    if commit is not None:
        meta.attrs["git_commit"] = commit
    config = results.metadata.get("config")
    meta.attrs["config_json"] = _config_to_json(config)
    if isinstance(config, SimulationConfig):
        meta.attrs["seed"] = config.seed
        meta.attrs["mode"] = config.mode.value
        meta.attrs["algorithm"] = config.algorithm.value
    else:
        seed = results.metadata.get("seed")
        if isinstance(seed, int):
            meta.attrs["seed"] = seed
        for key in ("mode", "algorithm"):
            value = results.metadata.get(key)
            if isinstance(value, str):
                meta.attrs[key] = value
    if "elapsed_seconds" in results.metadata:
        meta.attrs["elapsed_seconds"] = results.metadata["elapsed_seconds"]


def _read_metadata(meta: Any, path: Path) -> dict[str, object]:
    """Rebuild ``results.metadata`` from an HDF5 metadata group."""
    schema = _schema_version_of(meta, path)
    if schema >= 2:
        return _metadata_from_v2(meta, schema)
    return _metadata_from_v1(meta)


def _metadata_from_v2(meta: Any, schema: int) -> dict[str, object]:
    """Read a schema v2 metadata group.

    Numeric attributes are coerced to Python scalars (h5py returns numpy
    scalars, which ``json.dumps`` rejects downstream).
    """
    metadata: dict[str, object] = {"schema_version": schema}
    if "version" in meta.attrs:
        metadata["version"] = _as_str(meta.attrs["version"])
    if "seed" in meta.attrs:
        metadata["seed"] = int(meta.attrs["seed"])
    for key in ("mode", "algorithm", "git_commit"):
        if key in meta.attrs:
            metadata[key] = _as_str(meta.attrs[key])
    if "config_json" in meta.attrs:
        config = _config_from_json(_as_str(meta.attrs["config_json"]))
        if config is not None:
            metadata["config"] = config
    if "elapsed_seconds" in meta.attrs:
        metadata["elapsed_seconds"] = float(meta.attrs["elapsed_seconds"])
    return metadata


def _metadata_from_v1(meta: Any) -> dict[str, object]:
    """Read a legacy (mcising <= 0.23.0) metadata group.

    Legacy files carry only ``version``, ``config_json``, and
    ``elapsed_seconds``; the config is reconstructed best-effort so old
    files regain plot legends and correct per-site observables, and
    scalar provenance (seed, mode, algorithm) is derived from it.
    """
    metadata: dict[str, object] = {"schema_version": 1}
    # Documented legacy fallback: files from before the version stamp was
    # wired through (B12) genuinely recorded "unknown".
    metadata["version"] = _as_str(meta.attrs.get("version", "unknown"))
    if "config_json" in meta.attrs:
        config = _config_from_json(_as_str(meta.attrs["config_json"]))
        if config is not None:
            metadata["config"] = config
            metadata["seed"] = config.seed
            metadata["mode"] = config.mode.value
            metadata["algorithm"] = config.algorithm.value
    if "elapsed_seconds" in meta.attrs:
        metadata["elapsed_seconds"] = float(meta.attrs["elapsed_seconds"])
    return metadata


def _schema_version_of(meta: Any, path: Path) -> int:
    """Validate and return the metadata schema version of an open file.

    Files written by mcising <= 0.23.0 carry no ``schema_version``
    attribute and are schema 1.

    Raises
    ------
    ConfigurationError
        If the attribute is present but not a positive integer, or the
        schema is newer than this mcising supports — loading it would
        silently drop data the newer schema records.
    """
    if "schema_version" not in meta.attrs:
        return 1
    raw = meta.attrs["schema_version"]
    try:
        schema = int(raw)
    except (TypeError, ValueError):
        schema = 0
    if schema < 1:
        raise ConfigurationError(
            f"{path} has a corrupt schema_version attribute ({raw!r}); "
            "expected a positive integer."
        )
    if schema > HDF5_SCHEMA_VERSION:
        written_by = _as_str(meta.attrs.get("version", "a newer mcising"))
        raise ConfigurationError(
            f"{path} uses metadata schema {schema} (written by mcising "
            f"{written_by}), but this mcising ({package_version()}) "
            f"supports up to schema {HDF5_SCHEMA_VERSION}. Upgrade mcising "
            "to read it."
        )
    return schema


def _config_from_json(raw: str) -> SimulationConfig | None:
    """Best-effort reconstruction of a stored config record.

    Returns None when the record is empty or does not describe a valid
    config; loading data must not fail on a degraded provenance record
    (the resume path re-reads it loudly via ``_load_stored_config``).
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    try:
        return SimulationConfig.from_dict(data)
    except ConfigurationError:
        return None


def _as_str(value: object) -> str:
    """Decode an HDF5 string attribute (bytes in h5py 2.x-era files)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_temperature_group(f: Any, temp: float, results: SimulationResults) -> None:
    """Write a single temperature group to an HDF5 file handle."""
    group_name = f"T={temp:.6f}"
    grp = f.create_group(group_name)
    grp.attrs["temperature"] = temp
    # Added in P10 (additive attr, no schema bump): honest cluster-work
    # record for cluster algorithms; absent from older files.
    if temp in results.n_cluster_flips:
        grp.attrs["n_cluster_flips"] = results.n_cluster_flips[temp]

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

    if results.correlation_length is not None and temp in results.correlation_length:
        grp.create_dataset("correlation_length", data=results.correlation_length[temp])

    if (
        results.adaptive_diagnostics is not None
        and temp in results.adaptive_diagnostics
    ):
        diag = results.adaptive_diagnostics[temp]
        ad_grp = grp.create_group("adaptive_diagnostics")
        ad_grp.attrs["thermalization_sweeps"] = diag.thermalization_sweeps
        ad_grp.attrs["truncation_point"] = diag.truncation_point
        ad_grp.attrs["is_thermalized"] = diag.is_thermalized
        ad_grp.attrs["tau_int"] = diag.tau_int
        ad_grp.attrs["measurement_interval"] = diag.measurement_interval
        ad_grp.attrs["production_sweeps"] = diag.production_sweeps
        ad_grp.attrs["n_samples"] = diag.n_samples
        ad_grp.attrs["stationary_sweeps"] = diag.stationary_sweeps

    if temp in results.energy or temp in results.magnetization:
        _write_statistics_group(grp, results.statistics(temp))


def _write_statistics_group(grp: Any, stats: ObservableStatistics) -> None:
    """Write the derived per-temperature ``statistics`` subgroup (schema v3).

    Derived data for external inspection (``h5dump``, pandas): loading
    ignores this subgroup and recomputes statistics from the raw series,
    so the file never becomes a second source of truth. Non-finite
    values are omitted, never written as NaN/null (P07 policy).
    """
    st_grp = grp.create_group("statistics")
    st_grp.attrs["n_samples"] = stats.n_samples
    # Explicit convention label so the file is never ambiguous about
    # which chi it records (#39, P10).
    st_grp.attrs["susceptibility_kind"] = "connected"
    if math.isfinite(stats.tau_int):
        st_grp.attrs["tau_int"] = stats.tau_int
    for name, estimate in (
        ("energy", stats.energy),
        ("magnetization", stats.magnetization),
        ("abs_magnetization", stats.abs_magnetization),
        ("specific_heat", stats.specific_heat),
        ("susceptibility", stats.susceptibility),
        ("binder_cumulant", stats.binder_cumulant),
    ):
        if math.isfinite(estimate.value):
            st_grp.attrs[name] = estimate.value
        if math.isfinite(estimate.error):
            st_grp.attrs[f"{name}_error"] = estimate.error


def _config_to_json(config: object) -> str:
    """Serialize a config record to the JSON stored in ``config_json``.

    The resume-mismatch check compares both sides of a resume through
    this writer, so its output format is a compatibility surface: config
    dataclasses serialize via ``dataclasses.asdict`` (str-subclass enums
    emit their bare values, tuples become lists). Anything that cannot be
    serialized faithfully raises — a lossy provenance record is worse
    than a loud failure at write time.

    Raises
    ------
    ConfigurationError
        If ``config`` is not a config dataclass, a mapping, or None, or
        contains values JSON cannot represent.
    """
    if config is None:
        return "{}"
    if is_dataclass(config) and not isinstance(config, type):
        payload: dict[str, Any] = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        raise ConfigurationError(
            "A config record must be a config dataclass, a mapping, or "
            f"None, got {type(config).__name__}; refusing to write an "
            "unreadable provenance record."
        )
    try:
        return json.dumps(payload, indent=2)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Config record does not serialize to JSON: {exc}"
        ) from exc
