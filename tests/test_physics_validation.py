"""Physics validation tests using known analytical results.

These tests verify that the simulation produces physically correct results
for well-known cases of the 2D Ising model.
"""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, SimulationConfig
from mcising.constants import TC_SQUARE_2D
from mcising.simulation import Simulation


class TestMagnetizationTransition:
    """The 2D Ising model on a square lattice has a phase transition at
    T_c = 2/ln(1+sqrt(2)) ~ 2.269.

    Below T_c: spontaneous magnetization |m| > 0
    Above T_c: |m| → 0 (in thermodynamic limit)

    On finite lattices, the transition is broadened but still detectable.
    """

    def test_ordered_phase_below_tc(self) -> None:
        """Well below T_c, |m| should be close to 1."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=(1.5,),
            n_sweeps=500,
            n_thermalization=200,
            measurement_interval=5,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        mean_abs_m = float(np.mean(np.abs(results.magnetization[1.5])))
        assert mean_abs_m > 0.8

    def test_disordered_phase_above_tc(self) -> None:
        """Well above T_c, |m| should be small."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=(4.0,),
            n_sweeps=500,
            n_thermalization=200,
            measurement_interval=5,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        mean_abs_m = float(np.mean(np.abs(results.magnetization[4.0])))
        assert mean_abs_m < 0.3

    def test_magnetization_decreases_with_temperature(self) -> None:
        """<|m|> should decrease as T increases through T_c."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=(1.5, TC_SQUARE_2D, 4.0),
            n_sweeps=300,
            n_thermalization=200,
            measurement_interval=5,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)

        m_low = float(np.mean(np.abs(results.magnetization[1.5])))
        m_tc = float(np.mean(np.abs(results.magnetization[TC_SQUARE_2D])))
        m_high = float(np.mean(np.abs(results.magnetization[4.0])))

        assert m_low > m_tc > m_high


class TestEnergyBounds:
    """Energy per site for 2D square Ising with J1=1, J2=0, h=0
    is bounded: -2.0 <= E/N <= +2.0."""

    def test_energy_within_bounds(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        sim.metropolis_sweep(100, 0.5)
        e = sim.energy()
        assert -2.0 <= e <= 2.0

    def test_ground_state_energy(self) -> None:
        """All-up (or all-down) state has E/N = -2.0."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        spins = np.ones((8, 8), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.energy() == pytest.approx(-2.0)


class TestDetailedBalance:
    """Metropolis satisfies detailed balance, so at equilibrium:
    <E> and <M> should be statistically stationary."""

    def test_energy_stationarity(self) -> None:
        """After thermalization, energy should fluctuate around a mean."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        # Thermalize
        sim.metropolis_sweep(200, 0.5)

        # Collect energy measurements
        energies = []
        for _ in range(50):
            sim.metropolis_sweep(5, 0.5)
            energies.append(sim.energy())

        energies_arr = np.array(energies)
        # First half and second half should have similar means
        first_half = energies_arr[:25]
        second_half = energies_arr[25:]
        assert abs(np.mean(first_half) - np.mean(second_half)) < 0.3


class TestFieldEffect:
    """External field h should bias magnetization in its direction."""

    def test_positive_field_positive_magnetization(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 2.0, 42)  # h = 2.0
        sim.metropolis_sweep(500, 1.0)
        m = sim.magnetization()
        assert m > 0.5  # Strong field should align spins

    def test_negative_field_negative_magnetization(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, -2.0, 42)  # h = -2.0
        sim.metropolis_sweep(500, 1.0)
        m = sim.magnetization()
        assert m < -0.5
