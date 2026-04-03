"""Tests for energy computation via the Rust core."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation


class TestEnergyAllUp:
    """Analytical energy for the all-up state (J1=1, J2=0, h=0).

    For an L x L square lattice with all spins +1:
    E/N = -J1 * (coordination/2) = -1.0 * (4/2) = -2.0
    """

    def test_all_up_energy(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.energy() == pytest.approx(-2.0)

    def test_all_up_energy_larger(self) -> None:
        sim = IsingSimulation(16, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((16, 16), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.energy() == pytest.approx(-2.0)

    def test_all_down_energy(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = -np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.energy() == pytest.approx(-2.0)


class TestEnergyCheckerboard:
    """Checkerboard (antiferromagnetic) state has maximum NN frustration.

    For checkerboard with J1=1: every NN pair is anti-aligned, so
    E_nn/N = -J1 * sum_nn(s_i * s_j) / (2N) = +J1 * (coordination/2) = +2.0
    """

    def test_checkerboard_energy(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        sim.set_spins(spins)
        assert sim.energy() == pytest.approx(2.0)


class TestEnergyWithField:
    """Energy with external field h."""

    def test_all_up_with_field(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 1.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        # E/N = -2.0 (NN) + (-h * m) = -2.0 + (-1.0 * 1.0) = -3.0
        assert sim.energy() == pytest.approx(-3.0)

    def test_all_down_with_field(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 1.0, 42)
        spins = -np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        # E/N = -2.0 (NN) + (-h * m) = -2.0 + (-1.0 * (-1.0)) = -1.0
        assert sim.energy() == pytest.approx(-1.0)


class TestEnergyWithJ2:
    """Energy with next-nearest-neighbor coupling J2."""

    def test_all_up_with_j2(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.5, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        # E/N = -J1*(4/2) - J2*(4/2) = -2.0 - 1.0 = -3.0
        assert sim.energy() == pytest.approx(-3.0)

    def test_checkerboard_with_j2(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.5, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        sim.set_spins(spins)
        # NN: all anti-aligned -> +2.0
        # NNN: checkerboard -> all NNN aligned -> -J2*(4/2) = -1.0
        assert sim.energy() == pytest.approx(1.0)


class TestSpinEnergy:
    """Test per-spin energy computation."""

    def test_spin_energy_center_all_up(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        # All-up: each spin has local energy = -J1 * 4 = -4.0 (4 aligned NN)
        e = sim.spin_energy(1, 1)
        assert e == pytest.approx(-4.0)
