"""Tests for mid-run HDF5 checkpointing."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from mcising.config import LatticeConfig, SimulationConfig
from mcising.io import (
    checkpoint_run,
    init_checkpoint_file,
    load_completed_temperatures,
    save_temperature_group,
)
from mcising.simulation import Simulation, SimulationResults


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
        path = tmp_path / "ckpt.h5"
        results = SimulationResults(
            metadata={"version": "0.2.0", "config": None}
        )
        init_checkpoint_file(path, results)

        assert path.exists()
        with h5py.File(path, "r") as f:
            assert "metadata" in f
            assert f["metadata"].attrs["version"] == "0.2.0"
            # No temperature groups yet
            temp_groups = [k for k in f.keys() if k.startswith("T=")]
            assert len(temp_groups) == 0

    def test_save_temperature_group(
        self, sim: Simulation, tmp_path: Path
    ) -> None:
        path = tmp_path / "ckpt.h5"
        results = sim.run(show_progress=False)
        init_checkpoint_file(path, results)

        save_temperature_group(path, 3.0, results)

        with h5py.File(path, "r") as f:
            assert "T=3.000000" in f
            assert "energy" in f["T=3.000000"]
            # Other temps not yet written
            assert "T=2.000000" not in f

    def test_load_completed_temperatures(
        self, sim: Simulation, tmp_path: Path
    ) -> None:
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

        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        sim.sweep(10, 0.5)

        rng_state = sim.get_rng_state()
        spins = np.array(sim.get_spins())

        sim.sweep(5, 0.5)
        energy_original = sim.energy()

        # Restore and redo
        sim.set_spins(spins)
        sim.set_rng_state(rng_state)
        sim.sweep(5, 0.5)
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
    def test_creates_file_with_all_temps(
        self, sim: Simulation, tmp_path: Path
    ) -> None:
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
        results2 = checkpoint_run(
            sim2, path, show_progress=False, resume=True
        )

        # All temperatures should still be in results
        assert len(results2.temperatures) == 3
        # Resumed data for T=3.0 should match original
        assert np.allclose(results2.energy[3.0], original_energy_3)

    def test_roundtrip_matches_structure(
        self, sim: Simulation, tmp_path: Path
    ) -> None:
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
