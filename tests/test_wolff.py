"""Tests for the Wolff cluster algorithm."""

import numpy as np
import pytest
from mcising import Simulation, SimulationConfig
from mcising.config import AdaptiveConfig, Algorithm, LatticeConfig
from mcising.exceptions import ConfigurationError


class TestWolffConfig:
    """Test Wolff algorithm constraint enforcement."""

    def test_wolff_j2_nonzero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="J2=0, J3=0, and h=0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, j2=0.5),
                algorithm=Algorithm.WOLFF,
            )

    def test_wolff_h_nonzero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="J2=0, J3=0, and h=0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, h=1.0),
                algorithm=Algorithm.WOLFF,
            )

    def test_wolff_j2_and_h_nonzero_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            SimulationConfig(
                lattice=LatticeConfig(size=4, j2=0.5, h=1.0),
                algorithm=Algorithm.WOLFF,
            )

    def test_wolff_j2_zero_h_zero_ok(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4, j1=1.0, j2=0.0, h=0.0),
            algorithm=Algorithm.WOLFF,
        )
        assert config.algorithm == Algorithm.WOLFF


class TestWolffSimulation:
    """Test Wolff algorithm via high-level Simulation."""

    def test_basic_run(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.WOLFF,
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
            algorithm=Algorithm.WOLFF,
            seed=42,
        )
        sim = Simulation(config)
        result = sim.sweep(2.269, n_sweeps=10)
        assert "energy" in result
        assert "magnetization" in result
        assert "acceptance_rate" in result
        # Wolff acceptance_rate = cluster_size / N, should be > 0
        assert result["acceptance_rate"] > 0

    def test_metropolis_agreement(self) -> None:
        """Wolff and Metropolis should produce statistically consistent results."""
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

        # Run Wolff
        wolff_config = SimulationConfig(
            lattice=lattice,
            algorithm=Algorithm.WOLFF,
            temperatures=(2.0,),
            n_sweeps=2000,
            n_thermalization=200,
            measurement_interval=5,
            seed=123,
        )
        wolff_results = Simulation(wolff_config).run(show_progress=False)

        # Compare mean energies — should agree within statistical error
        metro_e = np.mean(metro_results.energy[2.0])
        wolff_e = np.mean(wolff_results.energy[2.0])
        metro_se = np.std(metro_results.energy[2.0]) / np.sqrt(
            len(metro_results.energy[2.0])
        )
        wolff_se = np.std(wolff_results.energy[2.0]) / np.sqrt(
            len(wolff_results.energy[2.0])
        )
        combined_se = np.sqrt(metro_se**2 + wolff_se**2)

        assert abs(metro_e - wolff_e) < 5 * combined_se, (
            f"Energy disagreement: Metropolis={metro_e:.4f}, Wolff={wolff_e:.4f}, "
            f"5σ={5 * combined_se:.4f}"
        )

    def test_wolff_adaptive(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            algorithm=Algorithm.WOLFF,
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
            algorithm=Algorithm.WOLFF,
            temperatures=(2.269,),
            n_sweeps=100,
            seed=42,
        )
        r1 = Simulation(config).run(show_progress=False)
        r2 = Simulation(config).run(show_progress=False)
        np.testing.assert_array_equal(r1.energy[2.269], r2.energy[2.269])
