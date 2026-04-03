"""Tests for J3 (third-nearest-neighbor) coupling support."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import Algorithm, LatticeConfig, SimulationConfig
from mcising.exceptions import ConfigurationError
from mcising.simulation import Simulation


class TestJ3Energy:
    """Test energy computation with J3 coupling."""

    def test_all_up_j3_only(self) -> None:
        """J3=1 only: E = -J3*z_tnn/2 = -1*4/2 = -2.0 per site."""
        sim = IsingSimulation(4, 0.0, 0.0, 1.0, 0.0, 42)
        sim.set_spins(np.ones((4, 4), dtype=np.int8))
        assert abs(sim.energy() - (-2.0)) < 1e-10

    def test_all_up_j1_j3(self) -> None:
        """J1=1, J3=0.5: E = -(J1*4 + J3*4)/2 = -3.0 per site."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.5, 0.0, 42)
        sim.set_spins(np.ones((4, 4), dtype=np.int8))
        assert abs(sim.energy() - (-3.0)) < 1e-10

    def test_all_up_j1_j2_j3(self) -> None:
        """J1=1, J2=0.5, J3=0.25: E = -(4 + 2 + 1)/2 = -3.5 per site."""
        sim = IsingSimulation(4, 1.0, 0.5, 0.25, 0.0, 42)
        sim.set_spins(np.ones((4, 4), dtype=np.int8))
        assert abs(sim.energy() - (-3.5)) < 1e-10

    def test_all_up_j1_j2_j3_h(self) -> None:
        """J1=1, J2=0.5, J3=0.25, h=1: E = -3.5 - 1.0 = -4.5 per site."""
        sim = IsingSimulation(4, 1.0, 0.5, 0.25, 1.0, 42)
        sim.set_spins(np.ones((4, 4), dtype=np.int8))
        assert abs(sim.energy() - (-4.5)) < 1e-10


class TestJ3Simulation:
    """Test simulation behavior with J3 coupling."""

    def test_j3_energy_decreases_at_low_t(self) -> None:
        """Metropolis with J3 should lower energy at low T."""
        sim = IsingSimulation(8, 0.0, 0.0, 1.0, 0.0, 42)
        e_before = sim.energy()
        sim.sweep(100, 10.0)  # beta=10 → low T
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_j1j2j3_energy_decreases_at_low_t(self) -> None:
        """Full J1-J2-J3 Hamiltonian: energy decreases at low T."""
        sim = IsingSimulation(8, 1.0, 0.3, 0.2, 0.0, 42)
        e_before = sim.energy()
        sim.sweep(100, 10.0)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_j3_deterministic(self) -> None:
        """Same seed + J3 → identical results."""
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.5, 0.0, 123)
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.5, 0.0, 123)
        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)
        np.testing.assert_array_equal(sim1.get_spins(), sim2.get_spins())

    def test_j3_getter(self) -> None:
        """J3 getter returns the correct value."""
        sim = IsingSimulation(4, 1.0, 0.5, 0.3, 0.0, 42)
        assert sim.j3 == 0.3


class TestJ3Config:
    """Test configuration with J3 coupling."""

    def test_lattice_config_j3_default(self) -> None:
        """J3 defaults to 0.0."""
        lc = LatticeConfig()
        assert lc.j3 == 0.0

    def test_lattice_config_j3_set(self) -> None:
        lc = LatticeConfig(j3=0.5)
        assert lc.j3 == 0.5

    def test_lattice_config_j3_invalid(self) -> None:
        with pytest.raises(ValueError, match="j3"):
            LatticeConfig(j3=float("inf"))

    def test_cluster_algorithm_j3_nonzero_raises(self) -> None:
        """Wolff with J3≠0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="J3=0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, j3=0.5),
                algorithm=Algorithm.WOLFF,
            )

    def test_cluster_algorithm_j3_zero_ok(self) -> None:
        """Wolff with J3=0 should work fine."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=4, j3=0.0),
            algorithm=Algorithm.WOLFF,
            temperatures=(2.0,),
            n_sweeps=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert 2.0 in results.energy


class TestJ3HighLevelSimulation:
    """Test high-level Simulation class with J3."""

    def test_run_with_j3(self) -> None:
        """Full simulation run with J3 coupling."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=8, j1=1.0, j3=0.5),
            temperatures=(3.0, 2.0),
            n_sweeps=50,
            measurement_interval=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 2
        assert all(t in results.energy for t in results.temperatures)
