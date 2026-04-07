"""Cross-lattice coupling validation tests.

Systematically validates every lattice type × coupling combination to ensure
the generalized Metropolis tables, index formulas, and energy computations
work correctly for all coordination numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation

# ── Lattice definitions ────────────────────────────────────────────────
# (lattice_type, size, z_nn, z_nnn, z_tnn)

LATTICES = [
    ("square", 6, 4, 4, 4),
    ("triangular", 6, 6, 6, 6),
    ("chain", 20, 2, 2, 2),
    ("honeycomb", 6, 3, 6, 3),
    ("cubic", 4, 6, 12, 8),
]

# ── Coupling definitions ──────────────────────────────────────────────
# (name, j1, j2, j3, h)

COUPLINGS = [
    ("J1", 1.0, 0.0, 0.0, 0.0),
    ("J2", 0.0, 1.0, 0.0, 0.0),
    ("J3", 0.0, 0.0, 1.0, 0.0),
    ("J1+J2", 1.0, 1.0, 0.0, 0.0),
    ("J1+J3", 1.0, 0.0, 1.0, 0.0),
    ("J2+J3", 0.0, 1.0, 1.0, 0.0),
    ("J1+J2+J3", 1.0, 1.0, 1.0, 0.0),
    ("J1+H", 1.0, 0.0, 0.0, 1.0),
    ("All", 1.0, 1.0, 1.0, 1.0),
]


def _expected_allup_energy(
    z_nn: int, z_nnn: int, z_tnn: int, j1: float, j2: float, j3: float, h: float
) -> float:
    """Analytical energy per site for all-up configuration."""
    return -(j1 * z_nn + j2 * z_nnn + j3 * z_tnn) / 2.0 - h


def _make_energy_params() -> (
    list[tuple[str, str, int, float, float, float, float, float]]
):
    """Generate (id, lattice_type, size, j1, j2, j3, h, expected_E) tuples."""
    params = []
    for lt, sz, z_nn, z_nnn, z_tnn in LATTICES:
        for cname, j1, j2, j3, h in COUPLINGS:
            expected = _expected_allup_energy(z_nn, z_nnn, z_tnn, j1, j2, j3, h)
            test_id = f"{lt}-{cname}"
            params.append((test_id, lt, sz, j1, j2, j3, h, expected))
    return params


def _make_dynamics_params() -> (
    list[tuple[str, str, int, float, float, float, float]]
):
    """Generate (id, lattice_type, size, j1, j2, j3, h) for energy-decrease tests."""
    single_couplings = [
        ("J1", 1.0, 0.0, 0.0, 0.0),
        ("J2", 0.0, 1.0, 0.0, 0.0),
        ("J3", 0.0, 0.0, 1.0, 0.0),
        ("J1+H", 1.0, 0.0, 0.0, 0.5),
    ]
    params = []
    for lt, sz, *_ in LATTICES:
        for cname, j1, j2, j3, h in single_couplings:
            test_id = f"{lt}-{cname}"
            params.append((test_id, lt, sz, j1, j2, j3, h))
    return params


# ══════════════════════════════════════════════════════════════════════
# A. All-up energy validation (45 tests)
# ══════════════════════════════════════════════════════════════════════


class TestAllUpEnergy:
    """Verify E = -(J1*z_nn + J2*z_nnn + J3*z_tnn)/2 - h for all-up state."""

    @pytest.mark.parametrize(
        "test_id,lattice_type,size,j1,j2,j3,h,expected_e",
        _make_energy_params(),
        ids=[p[0] for p in _make_energy_params()],
    )
    def test_allup_energy(
        self,
        test_id: str,
        lattice_type: str,
        size: int,
        j1: float,
        j2: float,
        j3: float,
        h: float,
        expected_e: float,
    ) -> None:
        sim = IsingSimulation(
            size, j1, j2, j3, h, 42, "metropolis", lattice_type
        )
        shape = sim.get_spins().shape
        sim.set_spins(np.ones(shape, dtype=np.int8))
        actual = sim.energy()
        assert abs(actual - expected_e) < 1e-10, (
            f"{test_id}: expected E={expected_e}, got {actual}"
        )


# ══════════════════════════════════════════════════════════════════════
# B. Energy decrease at low T (20 tests)
# ══════════════════════════════════════════════════════════════════════


class TestEnergyDecrease:
    """Verify Metropolis lowers energy at beta=100 for each lattice×coupling."""

    @pytest.mark.parametrize(
        "test_id,lattice_type,size,j1,j2,j3,h",
        _make_dynamics_params(),
        ids=[p[0] for p in _make_dynamics_params()],
    )
    def test_energy_decreases(
        self,
        test_id: str,
        lattice_type: str,
        size: int,
        j1: float,
        j2: float,
        j3: float,
        h: float,
    ) -> None:
        sim = IsingSimulation(
            size, j1, j2, j3, h, 42, "metropolis", lattice_type
        )
        e_before = sim.energy()
        sim.sweep(200, 100.0)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10, (
            f"{test_id}: energy increased at low T: "
            f"{e_before:.6f} → {e_after:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════
# C. Determinism: full J1+J2+J3+H strategy on every lattice (5 tests)
# ══════════════════════════════════════════════════════════════════════


class TestCrossLatticeDeterminism:
    """Verify same seed → identical spins for the most complex strategy."""

    @pytest.mark.parametrize(
        "lattice_type,size",
        [(lt, sz) for lt, sz, *_ in LATTICES],
        ids=[lt for lt, *_ in LATTICES],
    )
    def test_deterministic_j1j2j3h(
        self, lattice_type: str, size: int
    ) -> None:
        sim1 = IsingSimulation(
            size, 1.0, 0.5, 0.3, 0.5, 123, "metropolis", lattice_type
        )
        sim2 = IsingSimulation(
            size, 1.0, 0.5, 0.3, 0.5, 123, "metropolis", lattice_type
        )
        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)
        np.testing.assert_array_equal(
            sim1.get_spins(),
            sim2.get_spins(),
            err_msg=f"{lattice_type}: J1+J2+J3+H not deterministic",
        )
