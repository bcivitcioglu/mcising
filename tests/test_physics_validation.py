"""Physics validation tests using known analytical results.

These tests verify that the simulation produces physically correct results
for well-known cases of the 2D Ising model.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, SimulationConfig
from mcising.constants import TC_SQUARE_2D
from mcising.simulation import Simulation

from tests._stats import (
    DEFAULT_SEEDS,
    assert_mean_above,
    assert_mean_below,
    assert_ordered_means,
    assert_over_seeds,
    assert_samples_agree,
    assert_within_sigma,
)

#: Descending ladder for sub-Tc runs. The cool-down ramp between rungs is
#: what anneals the system through Tc: a single-temperature config ramps
#: from INF_TEMP=100 and spends ~1 sweep below Tc, i.e. it quenches, and
#: a quench freezes into a two-domain-wall stripe (<|m|> ~ 0) often
#: enough that no threshold survives it.
ORDERED_LADDER: Final = (4.0, 3.0, 2.5, TC_SQUARE_2D, 2.0, 1.5)


class TestMagnetizationTransition:
    """The 2D Ising model on a square lattice has a phase transition at
    T_c = 2/ln(1+sqrt(2)) ~ 2.269.

    Below T_c: spontaneous magnetization |m| > 0
    Above T_c: |m| → 0 (in thermodynamic limit)

    On finite lattices, the transition is broadened but still detectable.
    """

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_ordered_phase_below_tc(self, seed: int) -> None:
        """Well below T_c, <|m|> approaches the Onsager spontaneous value.

        m_s(T) = (1 - sinh(2/T)**-4)**(1/8) = 0.9866 at T=1.5. The 0.8
        threshold is a regime boundary ~0.19 below that analytic value
        and an order of magnitude above the disordered-phase scale
        (~0.07 at L=16) — not a number read off a run.
        """
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=ORDERED_LADDER,
            n_sweeps=500,
            n_thermalization=200,
            measurement_interval=5,
            seed=seed,
        )
        results = Simulation(config).run(show_progress=False)
        abs_m = np.abs(results.magnetization[1.5])
        assert_mean_above(abs_m, 0.8, label=f"<|m|>(T=1.5, seed={seed})")

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_disordered_phase_above_tc(self, seed: int) -> None:
        """Well above T_c, <|m|> is small.

        At T=4.0 on L=16 the finite-size <|m|> is ~0.07; the 0.3
        threshold is ~4x that. No ladder needed: there is no
        metastability above Tc.
        """
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=(4.0,),
            n_sweeps=500,
            n_thermalization=200,
            measurement_interval=5,
            seed=seed,
        )
        results = Simulation(config).run(show_progress=False)
        abs_m = np.abs(results.magnetization[4.0])
        assert_mean_below(abs_m, 0.3, label=f"<|m|>(T=4.0, seed={seed})")

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_magnetization_decreases_with_temperature(self, seed: int) -> None:
        """<|m|> should decrease as T increases through T_c.

        The extra T=2.0 rung anneals the 1.5 point instead of quenching
        it (see ORDERED_LADDER); only 1.5, Tc, and 4.0 are asserted on.
        """
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=(4.0, TC_SQUARE_2D, 2.0, 1.5),
            n_sweeps=300,
            n_thermalization=200,
            measurement_interval=5,
            seed=seed,
        )
        results = Simulation(config).run(show_progress=False)
        assert_ordered_means(
            [
                ("<|m|>(T=1.5)", np.abs(results.magnetization[1.5])),
                ("<|m|>(T=Tc)", np.abs(results.magnetization[TC_SQUARE_2D])),
                ("<|m|>(T=4.0)", np.abs(results.magnetization[4.0])),
            ],
            increasing=False,
        )


class TestEnergyBounds:
    """Energy per site for 2D square Ising with J1=1, J2=0, h=0
    is bounded: -2.0 <= E/N <= +2.0."""

    def test_energy_within_bounds(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42)
        sim.sweep(100, 0.5)
        e = sim.energy()
        assert -2.0 <= e <= 2.0

    def test_ground_state_energy(self) -> None:
        """All-up (or all-down) state has E/N = -2.0."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((8, 8), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.energy() == pytest.approx(-2.0)


class TestHighTemperatureLimit:
    """As beta -> 0 the high-temperature expansion becomes exact."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_beta_zero_energy_matches_expansion(self, seed: int) -> None:
        """At beta=0.05, <E>/site = -2 tanh(beta) up to O(tanh^3).

        Square lattice, J1=1: <s_i s_j> = tanh(beta J) + O(tanh^3), two
        bonds per site, so <E>/site = -2 tanh(0.05) = -0.0999 with a
        correction ~2e-4 — far below the statistical error. The one
        two-sided analytic comparison in the suite.

        Sampling every 10 sweeps is deliberate: as beta -> 0 Metropolis
        accepts nearly every proposal, each sweep flips nearly every
        spin, and a global flip preserves E exactly (Z2 symmetry) — so
        the energy decorrelates only through the rare rejections
        (tau_int ~ 17 sweeps at beta=0.01). At beta=0.05 with 10-sweep
        spacing the samples are effectively independent.
        """
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, seed)
        sim.sweep(100, 0.05)
        energies = np.empty(200)
        for i in range(200):
            sim.sweep(10, 0.05)
            energies[i] = sim.energy()
        assert_within_sigma(
            energies,
            -2.0 * math.tanh(0.05),
            label=f"<E>/site(beta=0.05, seed={seed})",
        )


class TestStationarity:
    """At equilibrium, <E> is statistically stationary.

    (Renamed from TestDetailedBalance: this checks stationarity, a
    necessary consequence of detailed balance, not detailed balance
    itself — a true visit-histogram test lands in P04.)
    """

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_energy_stationarity(self, seed: int) -> None:
        """After thermalization, the two halves of the series agree."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, seed)
        sim.sweep(1000, 0.5)

        energies = np.empty(100)
        for i in range(100):
            sim.sweep(5, 0.5)
            energies[i] = sim.energy()

        assert_samples_agree(
            energies[:50],
            energies[50:],
            label_a=f"first half (seed={seed})",
            label_b=f"second half (seed={seed})",
        )


class TestFieldEffect:
    """External field h should bias magnetization in its direction."""

    @pytest.mark.statistical
    def test_positive_field_positive_magnetization(self) -> None:
        """h=+2 at T=1 aligns spins: <m> -> +1 (regime boundary 0.5)."""

        def check(seed: int) -> None:
            sim = IsingSimulation(8, 1.0, 0.0, 0.0, 2.0, seed)
            sim.sweep(500, 1.0)
            mags = np.empty(50)
            for i in range(50):
                sim.sweep(5, 1.0)
                mags[i] = sim.magnetization()
            assert_mean_above(mags, 0.5, label=f"<m>(h=+2, seed={seed})")

        assert_over_seeds(check)

    @pytest.mark.statistical
    def test_negative_field_negative_magnetization(self) -> None:
        """h=-2 at T=1 anti-aligns spins: <m> -> -1."""

        def check(seed: int) -> None:
            sim = IsingSimulation(8, 1.0, 0.0, 0.0, -2.0, seed)
            sim.sweep(500, 1.0)
            mags = np.empty(50)
            for i in range(50):
                sim.sweep(5, 1.0)
                mags[i] = sim.magnetization()
            assert_mean_below(mags, -0.5, label=f"<m>(h=-2, seed={seed})")

        assert_over_seeds(check)
