"""Tests for HDF5 and JSON I/O."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import mcising
import numpy as np
import pytest
from mcising._provenance import HDF5_SCHEMA_VERSION
from mcising.config import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.exceptions import ConfigurationError
from mcising.io import (
    _config_to_json,
    load_completed_temperatures,
    load_hdf5,
    save_hdf5,
    save_json_summary,
)
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


class TestNClusterFlips:
    """P10: honest cluster-work record, round-tripped as an additive attr."""

    def test_metropolis_reports_zero(self, sim_results) -> None:
        for temp in sim_results.temperatures:
            assert sim_results.n_cluster_flips[temp] == 0

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_wolff_counts_one_cluster_per_measured_sweep(
        self, mode: ExecutionMode
    ) -> None:
        # Wolff: exactly one cluster per sweep, thermalization excluded,
        # so the count equals the measured production sweeps.
        config = _small_config(algorithm=Algorithm.WOLFF, mode=mode)
        results = Simulation(config).run(show_progress=False)
        for temp in results.temperatures:
            n_measured = len(results.energy[temp]) * config.measurement_interval
            assert results.n_cluster_flips[temp] == n_measured

    def test_roundtrip_preserves_cluster_flips(self, tmp_path: Path) -> None:
        config = _small_config(algorithm=Algorithm.WOLFF)
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "wolff.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)
        assert loaded.n_cluster_flips == results.n_cluster_flips
        assert loaded.n_cluster_flips  # non-empty


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

    def test_legacy_elapsed_seconds_restored(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.h5"
        write_legacy_hdf5(path, elapsed_seconds=12.5)
        loaded = load_hdf5(path)
        assert loaded.metadata["elapsed_seconds"] == 12.5

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
            "correlation_interval",
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


class TestExactRoundTrip:
    """P12: bit-exact round-trips with pinned dtypes and shapes per lattice."""

    SHAPES = {
        LatticeType.SQUARE: (4, 4),
        LatticeType.TRIANGULAR: (4, 4),
        LatticeType.HONEYCOMB: (4, 4, 2),
        LatticeType.CUBIC: (4, 4, 4),
        LatticeType.CHAIN: (4,),
    }

    @pytest.mark.parametrize("lattice_type", list(LatticeType), ids=lambda t: t.value)
    def test_roundtrip_exact_per_lattice(
        self, lattice_type: LatticeType, tmp_path: Path
    ) -> None:
        config = _small_config(
            lattice=LatticeConfig(lattice_type=lattice_type, size=4)
        )
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "exact.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)

        n = config.n_sweeps // config.measurement_interval
        for temp in results.temperatures:
            for name in ("energy", "magnetization"):
                saved = getattr(results, name)[temp]
                back = getattr(loaded, name)[temp]
                assert back.dtype == np.float64
                assert back.shape == (n,)
                assert np.array_equal(back, saved)
            configs = loaded.configurations[temp]
            assert configs.dtype == np.int8
            assert configs.shape == (n, *self.SHAPES[lattice_type])
            assert np.array_equal(configs, results.configurations[temp])

    def test_roundtrip_temperature_attr_exact(self, tmp_path: Path) -> None:
        # 2.269 has no exact binary representation; the float64 attr (not
        # the 6-decimal group name) must be the source of truth on load.
        results = Simulation(_small_config(temperatures=(2.269,))).run(
            show_progress=False
        )
        path = tmp_path / "tc.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)
        assert loaded.temperatures == [2.269]
        assert 2.269 in loaded.energy


class TestLoadOrdering:
    """#48: temperature groups must be ordered numerically, not lexically."""

    def test_load_orders_temperatures_numerically_past_ten(
        self, tmp_path: Path
    ) -> None:
        results = Simulation(_small_config(temperatures=(10.0, 3.5, 2.0))).run(
            show_progress=False
        )
        path = tmp_path / "order.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)
        # Lexical order would give [10.0, 2.0, 3.5].
        assert loaded.temperatures == [2.0, 3.5, 10.0]
        for temp in (10.0, 3.5, 2.0):
            assert np.array_equal(loaded.energy[temp], results.energy[temp])


class TestJsonSummaryValues:
    """P12: the JSON summary quotes the same numbers statistics() computes."""

    def test_summary_values_match_statistics(self, tmp_path: Path) -> None:
        results = Simulation(
            _small_config(n_sweeps=400, lattice=LatticeConfig(size=8))
        ).run(show_progress=False)
        path = tmp_path / "summary.json"
        save_json_summary(results, path)
        data = json.loads(path.read_text())

        assert data["temperatures"] == results.temperatures
        assert data["elapsed_seconds"] == results.metadata["elapsed_seconds"]
        for temp in results.temperatures:
            entry = data["results"][f"{temp:.6f}"]
            stats = results.statistics(temp)
            assert entry["mean_energy"] == float(np.mean(results.energy[temp]))
            assert entry["std_energy"] == float(np.std(results.energy[temp]))
            assert entry["energy_error"] == stats.energy.error
            assert entry["mean_abs_magnetization"] == float(
                np.mean(np.abs(results.magnetization[temp]))
            )
            assert entry["std_magnetization"] == float(
                np.std(results.magnetization[temp])
            )
            assert entry["abs_magnetization_error"] == stats.abs_magnetization.error
            assert entry["n_samples"] == stats.n_samples
            assert entry["tau_int"] == stats.tau_int
            assert entry["specific_heat"] == stats.specific_heat.value
            assert entry["specific_heat_error"] == stats.specific_heat.error
            assert entry["susceptibility"] == stats.susceptibility.value
            assert entry["susceptibility_error"] == stats.susceptibility.error
            assert entry["binder_cumulant"] == stats.binder_cumulant.value
            assert entry["susceptibility_kind"] == "connected"

    def test_summary_mean_correlation_length(self, tmp_path: Path) -> None:
        results = Simulation(_small_config(compute_correlation=True)).run(
            show_progress=False
        )
        path = tmp_path / "summary.json"
        save_json_summary(results, path)
        data = json.loads(path.read_text())
        assert results.correlation_length is not None
        for temp in results.temperatures:
            entry = data["results"][f"{temp:.6f}"]
            assert entry["mean_correlation_length"] == float(
                np.mean(results.correlation_length[temp])
            )


class TestCorrelationRoundTrip:
    """P12: correlation datasets round-trip exactly."""

    def test_roundtrip_correlation_exact(self, tmp_path: Path) -> None:
        results = Simulation(_small_config(compute_correlation=True)).run(
            show_progress=False
        )
        path = tmp_path / "corr.h5"
        save_hdf5(results, path)
        loaded = load_hdf5(path)

        assert results.correlation_function is not None
        assert results.correlation_length is not None
        assert loaded.correlation_function is not None
        assert loaded.correlation_length is not None
        for temp in results.temperatures:
            saved_d, saved_c = results.correlation_function[temp]
            back_d, back_c = loaded.correlation_function[temp]
            assert back_d.dtype == saved_d.dtype
            assert back_c.dtype == saved_c.dtype
            assert np.array_equal(back_d, saved_d)
            assert np.array_equal(back_c, saved_c)
            xi_saved = results.correlation_length[temp]
            xi_back = loaded.correlation_length[temp]
            assert xi_back.dtype == xi_saved.dtype
            assert np.array_equal(xi_back, xi_saved)


class TestErrorPaths:
    """P12: corrupt, truncated, malformed, and unwritable files fail loudly.

    h5py error messages vary across versions; these tests assert
    exception types only.
    """

    def test_corrupt_file_raises_oserror_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.h5"
        path.write_bytes(b"this is not an HDF5 file")
        with pytest.raises(OSError):
            load_hdf5(path)

    def test_corrupt_file_raises_oserror_on_scan(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.h5"
        path.write_bytes(b"this is not an HDF5 file")
        with pytest.raises(OSError):
            load_completed_temperatures(path)

    def test_truncated_file_raises_oserror(self, sim_results, tmp_path: Path) -> None:
        path = tmp_path / "trunc.h5"
        save_hdf5(sim_results, path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        with pytest.raises(OSError):
            load_hdf5(path)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_hdf5(tmp_path / "nope.h5")

    def test_temperature_group_without_attr_raises(
        self, sim_results, tmp_path: Path
    ) -> None:
        path = tmp_path / "noattr.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "a") as f:
            del f["T=3.000000"].attrs["temperature"]
        with pytest.raises(KeyError):
            load_hdf5(path)

    def test_file_without_temperature_groups_loads_empty(
        self, tmp_path: Path
    ) -> None:
        from mcising.simulation import SimulationResults

        path = tmp_path / "meta_only.h5"
        save_hdf5(SimulationResults(metadata={"config": None}), path)
        loaded = load_hdf5(path)
        assert loaded.temperatures == []
        assert loaded.energy == {}

    def test_save_hdf5_unwritable_path_raises(
        self, sim_results, tmp_path: Path
    ) -> None:
        # Cross-platform: the target's parent is an existing *file*.
        blocker = tmp_path / "blocker"
        blocker.write_text("in the way")
        with pytest.raises(OSError):
            save_hdf5(sim_results, blocker / "out.h5")

    def test_save_json_unwritable_path_raises(
        self, sim_results, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("in the way")
        with pytest.raises(OSError):
            save_json_summary(sim_results, blocker / "out.json")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_save_to_readonly_dir_raises(self, sim_results, tmp_path: Path) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root ignores directory write bits")
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        try:
            with pytest.raises(OSError):
                save_hdf5(sim_results, ro / "out.h5")
        finally:
            ro.chmod(0o700)

    def test_save_hdf5_overwrites_existing_file(
        self, sim_results, tmp_path: Path
    ) -> None:
        # Pinned policy: save_hdf5 replaces an existing file silently
        # (only checkpoint_run guards against accidental overwrite).
        path = tmp_path / "overwrite.h5"
        path.write_text("junk from an earlier era")
        save_hdf5(sim_results, path)
        loaded = load_hdf5(path)
        assert loaded.temperatures == [2.0, 3.0]


class TestConfigRecordTolerance:
    """P12: degraded provenance records load tolerantly; resume re-reads loudly."""

    def test_bad_enum_in_config_json_loads_without_config(
        self, sim_results, tmp_path: Path
    ) -> None:
        # Best-effort contract (_config_from_json): an invalid stored
        # record yields no config rather than a load failure; the resume
        # path surfaces it as a loud mismatch instead.
        path = tmp_path / "bad_enum.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "a") as f:
            raw = json.loads(f["metadata"].attrs["config_json"])
            raw["mode"] = "bogus"
            f["metadata"].attrs["config_json"] = json.dumps(raw)
        loaded = load_hdf5(path)
        assert "config" not in loaded.metadata
        assert np.array_equal(loaded.energy[3.0], sim_results.energy[3.0])

    def test_malformed_config_json_loads_without_config(
        self, sim_results, tmp_path: Path
    ) -> None:
        path = tmp_path / "malformed.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "a") as f:
            f["metadata"].attrs["config_json"] = "{not json"
        loaded = load_hdf5(path)
        assert "config" not in loaded.metadata

    def test_config_to_json_accepts_mapping(self) -> None:
        payload = json.loads(_config_to_json({"seed": 7, "mode": "cooldown"}))
        assert payload == {"seed": 7, "mode": "cooldown"}

    def test_bytes_attr_decoded_on_load(self, sim_results, tmp_path: Path) -> None:
        # h5py 2.x-era files store string attrs as bytes.
        path = tmp_path / "bytes.h5"
        save_hdf5(sim_results, path)
        with h5py.File(path, "a") as f:
            del f["metadata"].attrs["version"]
            f["metadata"].attrs["version"] = np.bytes_(b"0.99.0")
        loaded = load_hdf5(path)
        assert loaded.metadata["version"] == "0.99.0"

    def test_scalar_provenance_fallback_without_config(self, tmp_path: Path) -> None:
        # No config object: seed/mode/algorithm are taken from the
        # results metadata scalars when correctly typed.
        from mcising.simulation import SimulationResults

        results = SimulationResults(
            metadata={
                "config": None,
                "seed": 7,
                "mode": "cooldown",
                "algorithm": "wolff",
            }
        )
        path = tmp_path / "scalars.h5"
        save_hdf5(results, path)
        with h5py.File(path, "r") as f:
            attrs = f["metadata"].attrs
            assert int(attrs["seed"]) == 7
            assert attrs["mode"] == "cooldown"
            assert attrs["algorithm"] == "wolff"
