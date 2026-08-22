"""Tests for the Wolff cluster algorithm."""

import numpy as np
import pytest
from mcising import Simulation, SimulationConfig
from mcising._core import (
    IsingSimulation,
    run_independent_temperatures,
    run_parallel_tempering,
)
from mcising.config import AdaptiveConfig, Algorithm, LatticeConfig
from mcising.exceptions import ConfigurationError

from tests._stats import DEFAULT_SEEDS, assert_samples_agree


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

    def test_wolff_negative_j1_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="require J1>0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, j1=-1.0),
                algorithm=Algorithm.WOLFF,
            )

    def test_wolff_zero_j1_raises(self) -> None:
        # J1=0 gives bond probability 0: the "cluster" degenerates exactly
        # like J1<0 does, so it is rejected by the same guard.
        with pytest.raises(ConfigurationError, match="require J1>0"):
            SimulationConfig(
                lattice=LatticeConfig(size=4, j1=0.0),
                algorithm=Algorithm.WOLFF,
            )

    def test_wolff_negative_j1_message_names_the_alternative(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            SimulationConfig(
                lattice=LatticeConfig(size=4, j1=-1.0),
                algorithm=Algorithm.WOLFF,
            )
        message = str(exc_info.value)
        assert "use metropolis for antiferromagnetic couplings" in message
        assert "sublattice mapping is future work" in message


class TestClusterCouplingSignBoundary:
    """The J1>0 cluster guard fires at every boundary a user can reach.

    At J1<=0 the Fortuin-Kasteleyn bond probability 1 - exp(-2*beta*J1)
    is <= 0, so cluster growth never adds a site and Wolff/Swendsen-Wang
    silently degenerate into random single spin flips (B1). The previous
    guard was a debug_assert that vanished in release builds.
    """

    def test_run_with_negative_j1_raises(self) -> None:
        # The roadmap gate: Simulation(algorithm=WOLFF, j1=-1).run() raises
        # ConfigurationError. It fires at config construction, before any
        # sweep can sample the wrong ensemble.
        with pytest.raises(ConfigurationError, match="require J1>0"):
            Simulation(
                SimulationConfig(
                    lattice=LatticeConfig(size=4, j1=-1.0),
                    algorithm=Algorithm.WOLFF,
                )
            ).run(show_progress=False)

    def test_core_constructor_rejects_negative_j1(self) -> None:
        # Defense in depth for direct _core use, which bypasses
        # SimulationConfig. Rust boundary errors are ValueError; the P11
        # unification made ConfigurationError a ValueError subclass, so
        # `except ValueError` covers both layers.
        with pytest.raises(ValueError, match="requires J1>0"):
            IsingSimulation(4, -1.0, 0.0, 0.0, 0.0, 42, "wolff", "square")

    def test_independent_runner_rejects_negative_j1(self) -> None:
        # Before P04 this path would panic inside the Rayon closure
        # (PanicException), not raise.
        with pytest.raises(ValueError, match="requires J1>0"):
            run_independent_temperatures(
                4, -1.0, 0.0, 0.0, 0.0, 42, "wolff", "square", [2.0], 10, 10, 1
            )

    def test_parallel_tempering_rejects_negative_j1(self) -> None:
        with pytest.raises(ValueError, match="requires J1>0"):
            run_parallel_tempering(
                4,
                -1.0,
                0.0,
                0.0,
                0.0,
                42,
                "swendsen_wang",
                "square",
                [2.0, 2.5],
                10,
                10,
                1,
            )


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
        result = sim.sweep(10, temperature=2.269)
        assert "energy" in result
        assert "magnetization" in result
        assert "acceptance_rate" in result
        # Wolff acceptance_rate = cluster_size / N, should be > 0
        assert result["acceptance_rate"] > 0

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_metropolis_agreement(self, seed: int) -> None:
        """Wolff and Metropolis sample the same equilibrium ensemble at T=2.

        Metropolis gets a cool-down ladder: a single-temperature config
        ramps from INF_TEMP in ~1 sweep below Tc, so it can sit in a
        stripe state for hundreds of production sweeps and bias <E> by
        up to +0.25/site. Wolff decorrelates in O(1) cluster updates and
        needs no ladder. Comparison uses blocking standard errors
        (autocorrelation-aware), plus an absolute floor so the test
        cannot pass vacuously if one run's error estimate explodes.
        """
        lattice = LatticeConfig(size=16, j1=1.0, j2=0.0, h=0.0)
        metro_results = Simulation(
            SimulationConfig(
                lattice=lattice,
                algorithm=Algorithm.METROPOLIS,
                temperatures=(3.0, 2.5, 2.0),
                n_sweeps=3000,
                n_thermalization=500,
                measurement_interval=10,
                seed=seed,
            )
        ).run(show_progress=False)
        wolff_results = Simulation(
            SimulationConfig(
                lattice=lattice,
                algorithm=Algorithm.WOLFF,
                temperatures=(2.0,),
                n_sweeps=2000,
                n_thermalization=200,
                measurement_interval=5,
                seed=seed + 1,  # independent xoshiro stream
            )
        ).run(show_progress=False)

        assert_samples_agree(
            metro_results.energy[2.0],
            wolff_results.energy[2.0],
            label_a=f"Metropolis <E> (seed={seed})",
            label_b=f"Wolff <E> (seed={seed + 1})",
        )
        # Power floor: ~3% of |E|/site at T=2 and ~10x the expected
        # combined error — catches vacuous sigma-passes.
        metro_e = float(np.mean(metro_results.energy[2.0]))
        wolff_e = float(np.mean(wolff_results.energy[2.0]))
        assert abs(metro_e - wolff_e) < 0.04

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
