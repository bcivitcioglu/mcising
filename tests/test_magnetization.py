"""Tests for magnetization computation via the Rust core."""

from __future__ import annotations

import numpy as np
import pytest

from mcising._core import IsingSimulation


class TestMagnetization:
    def test_all_up_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(1.0)

    def test_all_down_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        spins = -np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(-1.0)

    def test_checkerboard_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(0.0)

    def test_magnetization_range(self, small_sim: IsingSimulation) -> None:
        """Random initial state has |m| <= 1."""
        m = small_sim.magnetization()
        assert -1.0 <= m <= 1.0

    def test_single_flip_changes_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(1.0)

        sim.flip_spin(0, 0)
        # One spin flipped: m = (16 - 2) / 16 = 14/16 = 0.875
        assert sim.magnetization() == pytest.approx(0.875)