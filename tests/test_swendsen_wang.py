"""Tests for the Swendsen-Wang cluster algorithm."""

import numpy as np
import pytest
from mcising import Simulation, SimulationConfig
from mcising.config import AdaptiveConfig, Algorithm, LatticeConfig
from mcising.exceptions import ConfigurationError


class TestSwendsenWangConfig:
    """Test Swendsen-Wang algorithm constraint enforcement."""

    def test_sw_j2_nonzero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="J2=0 and h=0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, j2=0.5),
                algorithm=Algorithm.SWENDSEN_WANG,
            )

    def test_sw_h_nonzero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="J2=0 and h=0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, h=1.0),
                algorithm=Algorithm.SWENDSEN_WANG,
            )

    def test_sw_j2_zero_h_zero_ok(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4, j1=1.0, j2=0.0, h=0.0),
            algorithm=Algorithm.SWENDSEN_WANG,
        )
        assert config.algorithm == Algorithm.SWENDSEN_WANG


class TestSwendsenWangSimulation:
    """Test Swendsen-Wang algorithm via high-level Simulation."""

    def test_basic_run(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.SWENDSEN_WANG,
            temperatures=(3.0, 2.269, 1.5),
            n_sweeps=200,
            n_thermalization=50,
            seed=42,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 3
        for t in results.temperatures:
            assert t in results.energy
            assert t in results.magnetization

    def test_sweep_method(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.SWENDSEN_WANG,
            seed=42,
        )
        sim = Simulation(config)
        result = sim.sweep(2.269, n_sweeps=10)
        assert "energy" in result
        assert "magnetization" in result
        assert "acceptance_rate" in result

    def test_metropolis_agreement(self) -> None:
        """SW and Metropolis should produce statistically consistent results."""
        lattice = LatticeConfig(size=16, j1=1.0, j2=0.0, h=0.0)

        # Run Metropolis
        metro_config = SimulationConfig(
            lattice=lattice,
            algorithm=Algorithm.METROPOLIS,
            temperatures=(2.0,),
            n_sweeps=5000,
            n_thermalization=500,
            measurement_interval=10,
            seed=42,
        )
        metro_results = Simulation(metro_config).run(show_progress=False)

        # Run Swendsen-Wang
        sw_config = SimulationConfig(
            lattice=lattice,
            algorithm=Algorithm.SWENDSEN_WANG,
            temperatures=(2.0,),
            n_sweeps=2000,
            n_thermalization=200,
            measurement_interval=5,
            seed=123,
        )
        sw_results = Simulation(sw_config).run(show_progress=False)

        # Compare mean energies
        metro_e = np.mean(metro_results.energy[2.0])
        sw_e = np.mean(sw_results.energy[2.0])
        metro_se = np.std(metro_results.energy[2.0]) / np.sqrt(
            len(metro_results.energy[2.0])
        )
        sw_se = np.std(sw_results.energy[2.0]) / np.sqrt(len(sw_results.energy[2.0]))
        combined_se = np.sqrt(metro_se**2 + sw_se**2)

        assert abs(metro_e - sw_e) < 5 * combined_se, (
            f"Energy disagreement: Metropolis={metro_e:.4f}, SW={sw_e:.4f}, "
            f"5σ={5 * combined_se:.4f}"
        )

    def test_sw_adaptive(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            algorithm=Algorithm.SWENDSEN_WANG,
            temperatures=(2.269,),
            adaptive=AdaptiveConfig(enabled=True, min_independent_samples=50),
            seed=42,
        )
        results = Simulation(config).run(show_progress=False)
        assert results.adaptive_diagnostics is not None
        diag = results.adaptive_diagnostics[2.269]
        assert diag.n_samples >= 1
        assert diag.tau_int > 0

    def test_deterministic(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.SWENDSEN_WANG,
            temperatures=(2.269,),
            n_sweeps=100,
            seed=42,
        )
        r1 = Simulation(config).run(show_progress=False)
        r2 = Simulation(config).run(show_progress=False)
        np.testing.assert_array_equal(r1.energy[2.269], r2.energy[2.269])
