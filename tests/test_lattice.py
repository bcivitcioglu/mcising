"""Tests for lattice construction and properties via the Rust core."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation


class TestLatticeInitialization:
    def test_creates_correct_size(self, small_sim: IsingSimulation) -> None:
        assert small_sim.lattice_size == 4
        assert small_sim.num_sites == 16

    def test_spins_shape(self, small_sim: IsingSimulation) -> None:
        spins = small_sim.get_spins()
        assert spins.shape == (4, 4)
        assert spins.dtype == np.int8

    def test_spins_are_plus_minus_one(self, small_sim: IsingSimulation) -> None:
        spins = small_sim.get_spins()
        unique = np.unique(spins)
        assert all(v in (-1, 1) for v in unique)

    def test_invalid_lattice_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Lattice size must be >= 2"):
            IsingSimulation(1, 1.0, 0.0, 0.0, 42)

    def test_invalid_j1_raises(self) -> None:
        with pytest.raises(ValueError, match="j1"):
            IsingSimulation(4, float("inf"), 0.0, 0.0, 42)

    def test_properties_accessible(self, small_sim: IsingSimulation) -> None:
        assert small_sim.j1 == 1.0
        assert small_sim.j2 == 0.0
        assert small_sim.h == 0.0


class TestSpinManipulation:
    def test_flip_spin(self, small_sim: IsingSimulation) -> None:
        original = small_sim.get_spins().copy()
        small_sim.flip_spin(0, 0)
        flipped = small_sim.get_spins()
        assert flipped[0, 0] == -original[0, 0]
        # Other spins unchanged
        assert np.array_equal(flipped[1:, :], original[1:, :])
        assert np.array_equal(flipped[0, 1:], original[0, 1:])

    def test_set_spins(self, small_sim: IsingSimulation) -> None:
        new_spins = np.ones((4, 4), dtype=np.int8)
        small_sim.set_spins(new_spins)
        result = small_sim.get_spins()
        assert np.array_equal(result, new_spins)

    def test_set_spins_wrong_shape_raises(self, small_sim: IsingSimulation) -> None:
        wrong = np.ones((3, 3), dtype=np.int8)
        with pytest.raises(ValueError, match="Expected shape"):
            small_sim.set_spins(wrong)

    def test_set_spins_invalid_values_raises(self, small_sim: IsingSimulation) -> None:
        invalid = np.zeros((4, 4), dtype=np.int8)  # 0 is not a valid spin
        with pytest.raises(ValueError, match="must be.*1"):
            small_sim.set_spins(invalid)

    def test_flip_out_of_bounds_raises(self, small_sim: IsingSimulation) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            small_sim.flip_spin(10, 10)
