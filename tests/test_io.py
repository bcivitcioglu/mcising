"""Tests for HDF5 and JSON I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mcising.config import LatticeConfig, SimulationConfig
from mcising.io import load_hdf5, save_hdf5, save_json_summary
from mcising.simulation import Simulation


@pytest.fixture
def sim_results():
    """Run a small simulation and return results."""
    config = SimulationConfig(
        lattice=LatticeConfig(size=4),
        temperatures=(3.0, 2.0),
        n_sweeps=20,
        measurement_interval=10,
    )
    sim = Simulation(config)
    return sim.run(show_progress=False)


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
        import json

        path = tmp_path / "summary.json"
        save_json_summary(sim_results, path)
        with open(path) as f:
            data = json.load(f)
        assert "temperatures" in data
        assert "results" in data
        assert len(data["results"]) == len(sim_results.temperatures)
