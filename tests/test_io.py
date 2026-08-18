"""Tests for HDF5 and JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import mcising
import numpy as np
import pytest
from mcising._provenance import HDF5_SCHEMA_VERSION
from mcising.config import ExecutionMode, LatticeConfig, SimulationConfig
from mcising.exceptions import ConfigurationError
from mcising.io import _config_to_json, load_hdf5, save_hdf5, save_json_summary
from mcising.simulation import Simulation

from tests._legacy_schema import write_legacy_hdf5

ALL_MODES = [
    ExecutionMode.COOLDOWN,
    ExecutionMode.INDEPENDENT,
    ExecutionMode.PARALLEL_TEMPERING,
]


def _small_config(**overrides):
    defaults = dict(
        lattice=LatticeConfig(size=4),
        temperatures=(3.0, 2.0),
        n_sweeps=20,
        measurement_interval=10,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


@pytest.fixture
def sim_results():
    """Run a small simulation and return results."""
    return Simulation(_small_config()).run(show_progress=False)


class TestHDF5:
    def test_save_creates_file(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "test.h5"
        save_hdf5(sim_results, path)
        assert path.exists()

    def test_roundtrip(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "test.h5"
        save_hdf5(sim_results, path)
        loaded = load_hdf5(path)

        assert len(loaded.temperatures) == len(sim_results.temperatures)
        for temp in sim_results.temperatures:
            assert temp in loaded.energy
            assert np.allclose(loaded.energy[temp], sim_results.energy[temp])
            assert np.allclose(
                loaded.magnetization[temp], sim_results.magnetization[temp]
            )

    def test_roundtrip_configurations(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "test.h5"
        save_hdf5(sim_results, path)
        loaded = load_hdf5(path)

        for temp in sim_results.temperatures:
            assert np.array_equal(
                loaded.configurations[temp], sim_results.configurations[temp]
            )

    def test_creates_parent_dirs(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "deep" / "test.h5"
        save_hdf5(sim_results, path)
        assert path.exists()


class TestJSON:
    def test_save_creates_file(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        save_json_summary(sim_results, path)
        assert path.exists()

    def test_json_has_temperatures(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        save_json_summary(sim_results, path)
        with open(path) as f:
            data = json.load(f)
        assert "temperatures" in data
        assert "results" in data
        assert len(data["results"]) == len(sim_results.temperatures)

    def test_has_provenance_fields(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        save_json_summary(sim_results, path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == mcising.__version__
        assert data["schema_version"] == HDF5_SCHEMA_VERSION
        assert data["seed"] == 42
        assert data["mode"] == "cooldown"
        assert data["algorithm"] == "metropolis"
        assert data["config"]["lattice"]["size"] == 4
        assert "elapsed_seconds" in data

    def test_summary_of_loaded_results_is_serializable(
        self, sim_results, tmp_path: Path
    ) -> None:
        # h5py hands back numpy scalars; the loader must coerce them or
        # json.dumps raises TypeError on the re-exported summary.
        h5 = tmp_path / "results.h5"
        save_hdf5(sim_results, h5)
        loaded = load_hdf5(h5)
        path = tmp_path / "summary.json"
        save_json_summary(loaded, path)
        with open(path) as f:
            data = json.load(f)
        assert data["seed"] == 42
        assert data["version"] == mcising.__version__


class TestProvenanceRoundTrip:
    """Gate: per-mode round-trip restores version and the config object."""

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
    def test_version_matches_package(self, mode: ExecutionMode, tmp_path: Path) -> None:
        config = _small_config(mode=mode)
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "results.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)
        assert loaded.metadata["version"] == mcising.__version__

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
    def test_config_restored_equals_original(
        self, mode: ExecutionMode, tmp_path: Path
    ) -> None:
        config = _small_config(mode=mode, seed=123, compute_correlation=True)
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "results.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)
        assert loaded.metadata["config"] == config

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
    def test_seed_mode_algorithm_roundtrip(
        self, mode: ExecutionMode, tmp_path: Path
    ) -> None:
        config = _small_config(mode=mode, seed=99)
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "results.h5"
        save_hdf5(results, path)

        with h5py.File(path, "r") as f:
            attrs = f["metadata"].attrs
            assert int(attrs["seed"]) == 99
            assert attrs["mode"] == mode.value
            assert attrs["algorithm"] == "metropolis"

        loaded = load_hdf5(path)
        assert loaded.metadata["seed"] == 99
        assert loaded.metadata["mode"] == mode.value
        assert loaded.metadata["algorithm"] == "metropolis"

    def test_schema_version_is_current(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "results.h5"
        save_hdf5(sim_results, path)
        loaded = load_hdf5(path)
        assert loaded.metadata["schema_version"] == HDF5_SCHEMA_VERSION

    def test_git_commit_absent_or_nonempty_str(
        self, sim_results, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.h5"
        save_hdf5(sim_results, path)
        loaded = load_hdf5(path)
        commit = loaded.metadata.get("git_commit")
        assert commit is None or (isinstance(commit, str) and commit)


class TestSchemaCompat:
    """Gate: old-schema files keep loading; newer schemas refuse loudly."""

    def test_legacy_file_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.h5"
        write_legacy_hdf5(path, temperatures=(3.0, 2.0), n_samples=5)
        loaded = load_hdf5(path)
        assert loaded.temperatures == [2.0, 3.0]
        assert loaded.energy[3.0].shape == (5,)
        assert loaded.metadata["version"] == "0.2.0"
        assert loaded.metadata["schema_version"] == 1

    def test_legacy_file_without_version_reports_unknown(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "legacy.h5"
        write_legacy_hdf5(path, version=None)
        loaded = load_hdf5(path)
        assert loaded.metadata["version"] == "unknown"

    def test_legacy_config_best_effort_restored(self, tmp_path: Path) -> None:
        config = _small_config(seed=7)
        path = tmp_path / "legacy.h5"
        write_legacy_hdf5(path, config_json=_config_to_json(config))
        loaded = load_hdf5(path)
        assert loaded.metadata["config"] == config
        assert loaded.metadata["seed"] == 7
        assert loaded.metadata["mode"] == "cooldown"
        assert loaded.metadata["algorithm"] == "metropolis"

    def test_legacy_file_without_config_json_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.h5"
        write_legacy_hdf5(path, config_json=None)
        loaded = load_hdf5(path)
        assert "config" not in loaded.metadata

    def test_future_schema_raises(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "future.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "a") as f:
            f["metadata"].attrs["schema_version"] = 99
        with pytest.raises(ConfigurationError, match="schema 99"):
            load_hdf5(path)

    def test_invalid_schema_version_raises(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "a") as f:
            f["metadata"].attrs["schema_version"] = "banana"
        with pytest.raises(ConfigurationError, match="schema_version"):
            load_hdf5(path)


class TestConfigJson:
    def test_none_serializes_to_empty_object(self) -> None:
        assert _config_to_json(None) == "{}"

    def test_shape_is_stable(self) -> None:
        # The resume-mismatch check compares configs through this writer;
        # a key or value-format change silently invalidates old
        # checkpoints, so the exact dict shape is pinned here.
        config = _small_config()
        data = json.loads(_config_to_json(config))
        assert set(data) == {
            "lattice",
            "algorithm",
            "seed",
            "temperatures",
            "n_sweeps",
            "n_thermalization",
            "measurement_interval",
            "compute_correlation",
            "store_configs",
            "adaptive",
            "mode",
            "swap_interval",
        }
        assert data["algorithm"] == "metropolis"
        assert data["mode"] == "cooldown"
        assert data["temperatures"] == [3.0, 2.0]
        assert set(data["lattice"]) == {"lattice_type", "size", "j1", "j2", "j3", "h"}
        assert data["lattice"]["lattice_type"] == "square"

    def test_unserializable_value_raises(self) -> None:
        config = _small_config()
        object.__setattr__(config, "seed", np.int64(3))
        with pytest.raises(ConfigurationError, match="serialize"):
            _config_to_json(config)

    def test_non_config_object_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="config record"):
            _config_to_json(42)


class TestDerivedObservablesFromLoadedFile:
    def test_specific_heat_matches_memory_with_store_configs_false(
        self, tmp_path: Path
    ) -> None:
        # Without stored configurations, a loaded file used to fall back
        # to num_sites=1, scaling Cv and chi by a factor of N (B12/B11
        # blast radius); the restored config re-enables the exact branch.
        config = _small_config(store_configs=False)
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "results.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)
        for temp in results.temperatures:
            assert loaded.specific_heat(temp) == pytest.approx(
                results.specific_heat(temp)
            )
            assert loaded.susceptibility(temp) == pytest.approx(
                results.susceptibility(temp)
            )


class TestStatisticsGroup:
    """Schema v3: derived statistics written per temperature, never read."""

    def test_schema_version_pin(self) -> None:
        # Deliberate pin: bumping the schema is a conscious decision
        # with a CHANGELOG entry, not a drive-by.
        assert HDF5_SCHEMA_VERSION == 3

    def test_statistics_attrs_match_recomputed(
        self, sim_results, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "r") as f:
            for temp in sim_results.temperatures:
                grp = f[f"T={temp:.6f}"]
                assert "statistics" in grp
                attrs = grp["statistics"].attrs
                stats = sim_results.statistics(temp)
                assert int(attrs["n_samples"]) == stats.n_samples
                assert float(attrs["energy"]) == stats.energy.value
                assert float(attrs["energy_error"]) == stats.energy.error
                assert (
                    float(attrs["specific_heat"]) == stats.specific_heat.value
                )
                assert (
                    float(attrs["binder_cumulant"])
                    == stats.binder_cumulant.value
                )

    def test_non_finite_attrs_omitted(self, sim_results, tmp_path: Path) -> None:
        # n=2 samples: jackknife errors are NaN by policy and must be
        # absent from the file, never stored as NaN.
        path = tmp_path / "results.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "r") as f:
            attrs = f["T=2.000000/statistics"].attrs
            assert "specific_heat" in attrs
            assert "specific_heat_error" not in attrs

    def test_loading_ignores_stored_statistics(
        self, sim_results, tmp_path: Path
    ) -> None:
        # The subgroup is written for external inspection only; loading
        # recomputes from the raw series, so a tampered value must have
        # no effect (no dual source of truth).
        path = tmp_path / "results.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "r+") as f:
            f["T=2.000000/statistics"].attrs["energy"] = 123.456
        loaded = load_hdf5(path)
        assert loaded.statistics(2.0).energy.value == pytest.approx(
            float(np.mean(sim_results.energy[2.0]))
        )

    def test_v2_file_without_statistics_group_loads(
        self, sim_results, tmp_path: Path
    ) -> None:
        # A schema-2 file (mcising 0.24.0) is exactly a v3 file without
        # the statistics subgroups.
        path = tmp_path / "v2.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "r+") as f:
            f["metadata"].attrs["schema_version"] = 2
            for temp in sim_results.temperatures:
                del f[f"T={temp:.6f}/statistics"]
        loaded = load_hdf5(path)
        assert loaded.metadata["schema_version"] == 2
        assert loaded.statistics(2.0).n_samples == 2


class TestJsonSummaryErrors:
    def test_error_fields_present_and_nan_omitted(self, tmp_path: Path) -> None:
        results = Simulation(
            _small_config(n_sweeps=400, lattice=LatticeConfig(size=8))
        ).run(show_progress=False)
        path = tmp_path / "summary.json"
        save_json_summary(results, path)
        with open(path) as f:
            entry = json.load(f)["results"]["2.000000"]
        assert entry["energy_error"] > 0.0
        assert entry["specific_heat_error"] > 0.0
        assert entry["binder_cumulant"] <= 2.0 / 3.0 + 1e-12
        assert entry["n_samples"] == 40

        short = Simulation(_small_config()).run(show_progress=False)
        short_path = tmp_path / "short.json"
        save_json_summary(short, short_path)
        with open(short_path) as f:
            entry = json.load(f)["results"]["2.000000"]
        # 2 samples: jackknife error is NaN -> key omitted, valid JSON.
        assert "specific_heat" in entry
        assert "specific_heat_error" not in entry
