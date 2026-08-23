"""Tests for mid-run HDF5 checkpointing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import mcising
import numpy as np
import pytest
from mcising._provenance import HDF5_SCHEMA_VERSION
from mcising.config import ExecutionMode, LatticeConfig, SimulationConfig
from mcising.exceptions import ConfigurationError
from mcising.io import (
    checkpoint_run,
    init_checkpoint_file,
    load_completed_temperatures,
    save_temperature_group,
)
from mcising.simulation import Simulation, SimulationResults

ALL_MODES = [
    ExecutionMode.COOLDOWN,
    ExecutionMode.INDEPENDENT,
    ExecutionMode.PARALLEL_TEMPERING,
]


@pytest.fixture
def small_config() -> SimulationConfig:
    return SimulationConfig(
        lattice=LatticeConfig(size=4),
        temperatures=(3.0, 2.0, 1.0),
        n_sweeps=20,
        measurement_interval=10,
    )


@pytest.fixture
def sim(small_config: SimulationConfig) -> Simulation:
    return Simulation(small_config)


class TestCheckpointPrimitives:
    def test_init_checkpoint_file(self, tmp_path: Path) -> None:
        # The writer derives version/schema itself (never trusts the
        # results object); run facts are omitted when no config is known.
        path = tmp_path / "ckpt.h5"
        results = SimulationResults(metadata={"config": None})
        init_checkpoint_file(path, results)

        assert path.exists()
        with h5py.File(path, "r") as f:
            assert "metadata" in f
            attrs = f["metadata"].attrs
            assert attrs["version"] == mcising.__version__
            assert int(attrs["schema_version"]) == HDF5_SCHEMA_VERSION
            assert attrs["config_json"] == "{}"
            assert "seed" not in attrs
            assert "mode" not in attrs
            assert "algorithm" not in attrs
            # No temperature groups yet
            temp_groups = [k for k in f.keys() if k.startswith("T=")]
            assert len(temp_groups) == 0

    def test_checkpoint_file_has_full_provenance(
        self, sim: Simulation, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        path = tmp_path / "ckpt.h5"
        checkpoint_run(sim, path, show_progress=False)

        with h5py.File(path, "r") as f:
            attrs = f["metadata"].attrs
            assert attrs["version"] == mcising.__version__
            assert int(attrs["schema_version"]) == HDF5_SCHEMA_VERSION
            assert int(attrs["seed"]) == small_config.seed
            assert attrs["mode"] == small_config.mode.value
            assert attrs["algorithm"] == small_config.algorithm.value
            assert attrs["config_json"] != "{}"

    def test_save_temperature_group(self, sim: Simulation, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.h5"
        results = sim.run(show_progress=False)
        init_checkpoint_file(path, results)

        save_temperature_group(path, 3.0, results)

        with h5py.File(path, "r") as f:
            assert "T=3.000000" in f
            assert "energy" in f["T=3.000000"]
            # Other temps not yet written
            assert "T=2.000000" not in f

    def test_load_completed_temperatures(self, sim: Simulation, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.h5"
        results = sim.run(show_progress=False)
        init_checkpoint_file(path, results)

        save_temperature_group(path, 3.0, results)
        save_temperature_group(path, 2.0, results)

        completed = load_completed_temperatures(path)
        assert completed == {3.0, 2.0}
        assert 1.0 not in completed


class TestOnTemperatureComplete:
    def test_callback_called_per_temperature(
        self, small_config: SimulationConfig
    ) -> None:
        sim = Simulation(small_config)
        recorded: list[float] = []

        def callback(temp: float, results: SimulationResults) -> None:
            recorded.append(temp)

        sim.run(show_progress=False, on_temperature_complete=callback)

        assert len(recorded) == 3
        # Temperatures are processed descending
        assert recorded == sorted(recorded, reverse=True)

    def test_skip_temperatures(self, small_config: SimulationConfig) -> None:
        sim = Simulation(small_config)
        recorded: list[float] = []

        def callback(temp: float, results: SimulationResults) -> None:
            recorded.append(temp)

        results = sim.run(
            show_progress=False,
            on_temperature_complete=callback,
            skip_temperatures=frozenset({3.0, 2.0}),
        )

        # Only T=1.0 should be simulated
        assert recorded == [1.0]
        assert 1.0 in results.energy
        assert 3.0 not in results.energy
        assert 2.0 not in results.energy


class TestRngState:
    def test_rng_state_roundtrip(self) -> None:
        """RNG state save/restore produces bitwise identical results."""
        from mcising._core import IsingSimulation

        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        sim.sweep(10, temperature=2.0)

        rng_state = sim.get_rng_state()
        spins = np.array(sim.get_spins())

        sim.sweep(5, temperature=2.0)
        energy_original = sim.energy()

        # Restore and redo
        sim.set_spins(spins)
        sim.set_rng_state(rng_state)
        sim.sweep(5, temperature=2.0)
        energy_restored = sim.energy()

        assert energy_original == energy_restored

    def test_checkpoint_preserves_simulation_state(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        """Checkpoint file contains sim_state group with spins and rng."""
        sim = Simulation(small_config)
        path = tmp_path / "ckpt.h5"
        checkpoint_run(sim, path, show_progress=False)

        with h5py.File(path, "r") as f:
            assert "sim_state" in f
            assert "spins" in f["sim_state"]
            assert "rng_state" in f["sim_state"]


class TestCheckpointRun:
    def test_creates_file_with_all_temps(self, sim: Simulation, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.h5"
        results = checkpoint_run(sim, path, show_progress=False)

        assert path.exists()
        completed = load_completed_temperatures(path)
        assert completed == {3.0, 2.0, 1.0}
        assert len(results.temperatures) == 3

    def test_checkpoint_interval(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        """With interval=2 and 3 temps, file has all 3 after run completes."""
        sim = Simulation(small_config)
        path = tmp_path / "ckpt.h5"
        checkpoint_run(sim, path, show_progress=False, checkpoint_interval=2)

        completed = load_completed_temperatures(path)
        assert completed == {3.0, 2.0, 1.0}

    def test_resume_skips_completed(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        path = tmp_path / "ckpt.h5"

        # First run: complete all temperatures
        sim1 = Simulation(small_config)
        results1 = checkpoint_run(sim1, path, show_progress=False)

        # Record original data for T=3.0
        original_energy_3 = results1.energy[3.0].copy()

        # Resume: all temps already done, should skip everything
        sim2 = Simulation(small_config)
        results2 = checkpoint_run(sim2, path, show_progress=False, resume=True)

        # All temperatures should still be in results
        assert len(results2.temperatures) == 3
        # Resumed data for T=3.0 should match original
        assert np.allclose(results2.energy[3.0], original_energy_3)

    def test_roundtrip_matches_structure(self, sim: Simulation, tmp_path: Path) -> None:
        """Checkpoint file is loadable via load_hdf5."""
        from mcising.io import load_hdf5

        path = tmp_path / "ckpt.h5"
        checkpoint_run(sim, path, show_progress=False)

        loaded = load_hdf5(path)
        assert len(loaded.temperatures) == 3
        for temp in [3.0, 2.0, 1.0]:
            assert temp in loaded.energy
            assert temp in loaded.magnetization
            assert temp in loaded.configurations


def _mode_config(mode: ExecutionMode) -> SimulationConfig:
    return SimulationConfig(
        lattice=LatticeConfig(size=4),
        temperatures=(3.0, 2.0, 1.0),
        n_sweeps=20,
        measurement_interval=10,
        mode=mode,
    )


class TestCheckpointAllModes:
    """checkpoint_run must produce a real file in every execution mode (B4).

    Before P06 the parallel modes silently ignored the checkpoint
    callback: no file was ever created while the CLI reported success.
    """

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_writes_all_temperature_groups(
        self, tmp_path: Path, mode: ExecutionMode
    ) -> None:
        path = tmp_path / "ckpt.h5"
        checkpoint_run(Simulation(_mode_config(mode)), path, show_progress=False)
        assert path.exists(), f"no checkpoint file written in {mode.value} mode"
        assert load_completed_temperatures(path) == {3.0, 2.0, 1.0}

    @pytest.mark.parametrize(
        "mode", [ExecutionMode.INDEPENDENT, ExecutionMode.PARALLEL_TEMPERING]
    )
    def test_callback_fires_once_per_temperature(self, mode: ExecutionMode) -> None:
        seen: list[float] = []
        Simulation(_mode_config(mode)).run(
            show_progress=False,
            on_temperature_complete=lambda t, _results: seen.append(t),
        )
        assert sorted(seen) == [1.0, 2.0, 3.0]


class TestResumeIndependent:
    """Independent-mode resume completes the scan with unchanged streams."""

    def test_resume_completes_scan_and_preserves_streams(self, tmp_path: Path) -> None:
        config = _mode_config(ExecutionMode.INDEPENDENT)
        full = Simulation(config).run(show_progress=False)

        # Forge an interrupted checkpoint: only T=3.0 and T=2.0 finished.
        path = tmp_path / "ckpt.h5"
        partial = Simulation(config).run(
            show_progress=False, skip_temperatures=frozenset({1.0})
        )
        init_checkpoint_file(path, partial)
        for temp in (3.0, 2.0):
            save_temperature_group(path, temp, partial)

        resumed = checkpoint_run(
            Simulation(config), path, show_progress=False, resume=True
        )

        assert load_completed_temperatures(path) == {3.0, 2.0, 1.0}
        assert set(resumed.temperatures) == {3.0, 2.0, 1.0}
        # The seed-offset contract: the resumed T=1.0 arrays are
        # byte-identical to the uninterrupted full run's.
        np.testing.assert_array_equal(resumed.energy[1.0], full.energy[1.0])
        np.testing.assert_array_equal(
            resumed.magnetization[1.0], full.magnetization[1.0]
        )


class TestResumeParallelTempering:
    """PT resume is all-or-nothing: the replicas form one coupled ensemble."""

    def test_fully_complete_ladder_resumes_from_file(self, tmp_path: Path) -> None:
        config = _mode_config(ExecutionMode.PARALLEL_TEMPERING)
        path = tmp_path / "ckpt.h5"
        first = checkpoint_run(Simulation(config), path, show_progress=False)

        resumed = checkpoint_run(
            Simulation(config), path, show_progress=False, resume=True
        )
        assert set(resumed.temperatures) == {3.0, 2.0, 1.0}
        for temp in (3.0, 2.0, 1.0):
            np.testing.assert_array_equal(resumed.energy[temp], first.energy[temp])

    def test_partial_ladder_raises(self, tmp_path: Path) -> None:
        config = _mode_config(ExecutionMode.PARALLEL_TEMPERING)
        full = Simulation(config).run(show_progress=False)

        path = tmp_path / "ckpt.h5"
        init_checkpoint_file(path, full)
        for temp in (3.0, 2.0):
            save_temperature_group(path, temp, full)

        with pytest.raises(ConfigurationError, match="coupled"):
            checkpoint_run(Simulation(config), path, show_progress=False, resume=True)


class TestResumeConfigGuard:
    """Resume refuses a checkpoint written with a different config."""

    @pytest.fixture
    def checkpoint(self, tmp_path: Path, small_config: SimulationConfig) -> Path:
        path = tmp_path / "ckpt.h5"
        checkpoint_run(Simulation(small_config), path, show_progress=False)
        return path

    def test_changed_j1_raises(
        self, checkpoint: Path, small_config: SimulationConfig
    ) -> None:
        changed = replace(small_config, lattice=LatticeConfig(size=4, j1=2.0))
        with pytest.raises(ConfigurationError, match="j1"):
            checkpoint_run(
                Simulation(changed), checkpoint, show_progress=False, resume=True
            )

    def test_changed_seed_raises(
        self, checkpoint: Path, small_config: SimulationConfig
    ) -> None:
        changed = replace(small_config, seed=7)
        with pytest.raises(ConfigurationError, match="seed"):
            checkpoint_run(
                Simulation(changed), checkpoint, show_progress=False, resume=True
            )

    def test_extended_temperatures_allowed(
        self, checkpoint: Path, small_config: SimulationConfig
    ) -> None:
        extended = replace(small_config, temperatures=(3.0, 2.0, 1.0, 0.5))
        results = checkpoint_run(
            Simulation(extended), checkpoint, show_progress=False, resume=True
        )
        assert set(results.temperatures) == {3.0, 2.0, 1.0, 0.5}
        assert load_completed_temperatures(checkpoint) == {3.0, 2.0, 1.0, 0.5}

    def test_identical_config_resumes(
        self, checkpoint: Path, small_config: SimulationConfig
    ) -> None:
        results = checkpoint_run(
            Simulation(small_config), checkpoint, show_progress=False, resume=True
        )
        assert set(results.temperatures) == {3.0, 2.0, 1.0}


class TestExistingCheckpointWithoutResume:
    """resume=False against an existing file refuses before any compute.

    The pre-P07 behavior was a provenance-corruption path: stale metadata
    kept, colliding temperature groups crashing after the sweeps ran, or
    two runs' ensembles silently merged into one file.
    """

    def test_raises_before_running(self, sim: Simulation, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.h5"
        init_checkpoint_file(path, SimulationResults(metadata={"config": None}))

        with pytest.raises(ConfigurationError, match="resume=True"):
            checkpoint_run(sim, path, show_progress=False)

        # Nothing ran and the file was not touched.
        assert load_completed_temperatures(path) == set()


class TestLegacyCheckpoint:
    """Files written by mcising <= 0.23.0 stay resumable and keep schema v1."""

    def test_resume_legacy_checkpoint_completes(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        from mcising.io import _config_to_json

        from tests._legacy_schema import write_legacy_hdf5

        path = tmp_path / "legacy.h5"
        write_legacy_hdf5(
            path,
            config_json=_config_to_json(small_config),
            temperatures=(3.0, 2.0),
        )

        results = checkpoint_run(
            Simulation(small_config), path, show_progress=False, resume=True
        )

        assert load_completed_temperatures(path) == {3.0, 2.0, 1.0}
        assert set(results.temperatures) == {3.0, 2.0, 1.0}
        # A file records the code that created it: resuming must not
        # upgrade the metadata group to schema v2 in place.
        with h5py.File(path, "r") as f:
            assert "schema_version" not in f["metadata"].attrs

    def test_future_schema_checkpoint_refuses_resume(
        self, sim: Simulation, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        path = tmp_path / "ckpt.h5"
        checkpoint_run(sim, path, show_progress=False)
        with h5py.File(path, "a") as f:
            f["metadata"].attrs["schema_version"] = 99

        with pytest.raises(ConfigurationError, match="schema 99"):
            checkpoint_run(
                Simulation(small_config), path, show_progress=False, resume=True
            )


class TestResumeRecordDegradation:
    """P12: resume guards on degraded or absent config records."""

    def _completed_checkpoint(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> Path:
        path = tmp_path / "ckpt.h5"
        checkpoint_run(Simulation(small_config), path, show_progress=False)
        return path

    def test_resume_unreadable_config_record_raises(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        path = self._completed_checkpoint(small_config, tmp_path)
        with h5py.File(path, "a") as f:
            f["metadata"].attrs["config_json"] = "{broken"
        with pytest.raises(ConfigurationError, match="unreadable"):
            checkpoint_run(
                Simulation(small_config), path, show_progress=False, resume=True
            )

    def test_resume_null_config_record_raises(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        # "null" parses as JSON but is not a dict: still unreadable.
        path = self._completed_checkpoint(small_config, tmp_path)
        with h5py.File(path, "a") as f:
            f["metadata"].attrs["config_json"] = "null"
        with pytest.raises(ConfigurationError, match="unreadable"):
            checkpoint_run(
                Simulation(small_config), path, show_progress=False, resume=True
            )

    def test_resume_bad_enum_in_record_raises_mismatch(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        # A bad enum string in the stored record parses as a dict, so it
        # surfaces as a config mismatch rather than an unreadable record.
        path = self._completed_checkpoint(small_config, tmp_path)
        with h5py.File(path, "a") as f:
            raw = json.loads(f["metadata"].attrs["config_json"])
            raw["mode"] = "bogus"
            f["metadata"].attrs["config_json"] = json.dumps(raw)
        with pytest.raises(ConfigurationError, match="does not match"):
            checkpoint_run(
                Simulation(small_config), path, show_progress=False, resume=True
            )

    def test_resume_without_config_record_is_unguarded(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        # Documented legacy tolerance: no record -> the mismatch guard
        # silently passes (it cannot see what it cannot read).
        path = self._completed_checkpoint(small_config, tmp_path)
        with h5py.File(path, "a") as f:
            del f["metadata"].attrs["config_json"]
        changed = replace(
            small_config, lattice=LatticeConfig(size=4, j1=0.5)
        )
        results = checkpoint_run(
            Simulation(changed), path, show_progress=False, resume=True
        )
        assert set(results.temperatures) == {3.0, 2.0, 1.0}

    def test_resume_without_metadata_group_is_unguarded(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        path = self._completed_checkpoint(small_config, tmp_path)
        with h5py.File(path, "a") as f:
            del f["metadata"]
        results = checkpoint_run(
            Simulation(small_config), path, show_progress=False, resume=True
        )
        assert set(results.temperatures) == {3.0, 2.0, 1.0}


class TestCheckpointTailFlush:
    def test_interval_larger_than_scan_creates_file_at_end(
        self, small_config: SimulationConfig, tmp_path: Path
    ) -> None:
        # checkpoint_interval > number of temperatures: nothing flushes
        # mid-run, so the tail flush must create the file itself.
        path = tmp_path / "tail.h5"
        checkpoint_run(
            Simulation(small_config),
            path,
            show_progress=False,
            checkpoint_interval=10,
        )
        with h5py.File(path, "r") as f:
            groups = {k for k in f if k.startswith("T=")}
        assert groups == {"T=3.000000", "T=2.000000", "T=1.000000"}


class TestResumeCorrelationMerge:
    def test_resumed_correlation_data_merged(self, tmp_path: Path) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(3.0, 2.0),
            n_sweeps=20,
            measurement_interval=10,
            compute_correlation=True,
        )
        path = tmp_path / "corr.h5"
        checkpoint_run(Simulation(config), path, show_progress=False)

        extended = replace(config, temperatures=(3.0, 2.0, 1.0))
        results = checkpoint_run(
            Simulation(extended), path, show_progress=False, resume=True
        )
        assert results.temperatures == [3.0, 2.0, 1.0]
        assert results.correlation_function is not None
        assert results.correlation_length is not None
        for temp in (3.0, 2.0, 1.0):
            assert temp in results.correlation_function
            assert temp in results.correlation_length
