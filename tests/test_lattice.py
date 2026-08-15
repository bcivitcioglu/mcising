"""Tests for lattice construction and properties via the Rust core."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import (
    IsingSimulation,
    run_independent_temperatures,
    run_parallel_tempering,
)
from mcising.config import LatticeConfig, LatticeType
from mcising.exceptions import ConfigurationError


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
            IsingSimulation(1, 1.0, 0.0, 0.0, 0.0, 42)

    def test_invalid_j1_raises(self) -> None:
        with pytest.raises(ValueError, match="j1"):
            IsingSimulation(4, float("inf"), 0.0, 0.0, 0.0, 42)

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
        with pytest.raises(ValueError, match="Expected 16 spins"):
            small_sim.set_spins(wrong)

    def test_set_spins_invalid_values_raises(self, small_sim: IsingSimulation) -> None:
        invalid = np.zeros((4, 4), dtype=np.int8)  # 0 is not a valid spin
        with pytest.raises(ValueError, match="must be.*1"):
            small_sim.set_spins(invalid)

    def test_flip_out_of_bounds_raises(self, small_sim: IsingSimulation) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            small_sim.flip_spin(10, 10)


class TestOddSizeBoundary:
    """The even-L guard for triangular/honeycomb fires at every boundary.

    Row-parity offset coordinates make rows 0 and L-1 share a parity when
    L is odd, so bonds across the vertical wrap seam are not reciprocal
    and the Hamiltonian is silently invalid (B2, #13). Correct odd-L
    periodic wraps are research-shaped; the configuration is rejected.
    """

    @pytest.mark.parametrize(
        "lattice_type", [LatticeType.TRIANGULAR, LatticeType.HONEYCOMB]
    )
    def test_config_rejects_odd_size(self, lattice_type: LatticeType) -> None:
        with pytest.raises(ConfigurationError, match="requires even size L"):
            LatticeConfig(lattice_type=lattice_type, size=5)

    @pytest.mark.parametrize(
        "lattice_type", [LatticeType.TRIANGULAR, LatticeType.HONEYCOMB]
    )
    def test_config_accepts_even_size(self, lattice_type: LatticeType) -> None:
        config = LatticeConfig(lattice_type=lattice_type, size=6)
        assert config.size == 6

    @pytest.mark.parametrize("lattice_name", ["triangular", "honeycomb"])
    def test_core_constructor_rejects_odd_size(self, lattice_name: str) -> None:
        # Defense in depth for direct _core use, which bypasses
        # LatticeConfig. Rust errors currently map to ValueError, not
        # ConfigurationError (unified in the P10/P11 API phases).
        with pytest.raises(ValueError, match="requires even size L"):
            IsingSimulation(5, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", lattice_name)

    def test_square_still_accepts_odd_size(self) -> None:
        # The guard is triangular/honeycomb-specific: square, chain, and
        # cubic wraps are parity-free and odd L remains valid.
        sim = IsingSimulation(5, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "square")
        assert sim.num_sites == 25

    def test_independent_runner_rejects_odd_size(self) -> None:
        with pytest.raises(ValueError, match="requires even size L"):
            run_independent_temperatures(
                5, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "triangular", [2.0], 10, 10, 1
            )

    def test_parallel_tempering_rejects_odd_size(self) -> None:
        with pytest.raises(ValueError, match="requires even size L"):
            run_parallel_tempering(
                5,
                1.0,
                0.0,
                0.0,
                0.0,
                42,
                "metropolis",
                "honeycomb",
                [2.0, 2.5],
                10,
                10,
                1,
            )
