"""Tests for the 3D cubic lattice."""

from __future__ import annotations

import numpy as np
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, LatticeType, SimulationConfig
from mcising.simulation import Simulation


class TestCubicEnergy:
    """Test energy computation on cubic lattice."""

    def test_all_up_energy(self) -> None:
        """All spins up: E = -J1 * 6 / 2 = -3.0 per site."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "cubic")
        n = sim.num_sites  # 4^3 = 64
        sim.set_spins(np.ones(n, dtype=np.int8))
        assert abs(sim.energy() - (-3.0)) < 1e-10

    def test_all_up_with_field(self) -> None:
        """All up, J1=1, h=1: E = -3.0 - 1.0 = -4.0 per site."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 1.0, 42, "metropolis", "cubic")
        sim.set_spins(np.ones(sim.num_sites, dtype=np.int8))
        assert abs(sim.energy() - (-4.0)) < 1e-10


class TestCubicSimulation:
    """Test simulation behavior on cubic lattice."""

    def test_energy_decreases_at_low_t(self) -> None:
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "cubic")
        e_before = sim.energy()
        sim.sweep(100, 10.0)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_deterministic(self) -> None:
        sim1 = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "cubic")
        sim2 = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "cubic")
        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)
        np.testing.assert_array_equal(sim1.get_spins(), sim2.get_spins())

    def test_spins_shape_is_3d(self) -> None:
        """Cubic should return a 3D spin array (L, L, L)."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "cubic")
        spins = sim.get_spins()
        assert spins.ndim == 3
        assert spins.shape == (4, 4, 4)

    def test_num_sites(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "cubic")
        assert sim.num_sites == 64


class TestCubicCluster:
    """Test cluster algorithms on cubic lattice."""

    def test_wolff_runs(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "cubic")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0

    def test_swendsen_wang_runs(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "swendsen_wang", "cubic")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0


class TestCubicHighLevel:
    """Test high-level Simulation class with cubic lattice."""

    def test_run_cubic(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.CUBIC,
                size=4,
                j1=1.0,
            ),
            temperatures=(5.0, 4.5, 3.0),
            n_sweeps=50,
            measurement_interval=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 3
        assert all(t in results.energy for t in results.temperatures)
