"""The closed-form oracles in ``tests/_analytic.py`` are themselves checked.

The Ferdinand-Fisher finite-torus solution is the reference for every
Wang-Landau physics gate on the square lattice, so it is pinned against a
brute-force enumeration of the 4 x 4 torus and against Onsager's infinite
lattice before it judges any sampler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from mcising.constants import TC_SQUARE_2D

from tests._analytic import (
    onsager_energy_per_site,
    square_torus_energy_per_site,
    square_torus_log_z,
    square_torus_specific_heat,
)


def _enumerate_square4() -> np.ndarray:
    """Total energies of all 65 536 states of the 4 x 4 torus (J = 1)."""
    size, n = 4, 16
    states = np.arange(1 << n, dtype=np.uint32)
    bits = ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
    grid = (2 * bits - 1).reshape(-1, size, size)
    bonds = (grid * np.roll(grid, 1, axis=1)).sum(axis=(1, 2))
    bonds = bonds + (grid * np.roll(grid, 1, axis=2)).sum(axis=(1, 2))
    return -bonds.astype(np.float64)


@pytest.mark.parametrize("temperature", [1.0, TC_SQUARE_2D, 4.0])
def test_ferdinand_fisher_matches_brute_force_on_the_4x4_torus(
    temperature: float,
) -> None:
    energies = _enumerate_square4()
    beta = 1.0 / temperature
    weights = np.exp(-beta * (energies - energies.min()))
    log_z = math.log(weights.sum()) - beta * energies.min()
    mean_e = (weights * energies).sum() / weights.sum()
    mean_e2 = (weights * energies * energies).sum() / weights.sum()
    assert square_torus_log_z(4, temperature) == pytest.approx(log_z, rel=1e-10)
    assert square_torus_energy_per_site(4, temperature) == pytest.approx(
        mean_e / 16, rel=1e-7
    )
    assert square_torus_specific_heat(4, temperature) == pytest.approx(
        beta * beta * (mean_e2 - mean_e * mean_e) / 16, rel=1e-5
    )


@pytest.mark.parametrize("temperature", [1.5, 2.0, 2.5, 3.0])
def test_ferdinand_fisher_converges_to_onsager(temperature: float) -> None:
    """Away from T_c the L = 64 torus is the infinite lattice to 1e-5."""
    assert square_torus_energy_per_site(64, temperature) == pytest.approx(
        onsager_energy_per_site(temperature), abs=1e-5
    )


def test_ferdinand_fisher_finite_size_offset_at_tc_shrinks_with_size() -> None:
    """At T_c the torus energy lies below -sqrt(2) by an offset ~ 0.6/L."""
    offsets = [
        square_torus_energy_per_site(size, TC_SQUARE_2D) + math.sqrt(2.0)
        for size in (8, 16, 32)
    ]
    assert all(offset < 0 for offset in offsets)
    assert abs(offsets[0]) > abs(offsets[1]) > abs(offsets[2])
    assert offsets[1] / offsets[2] == pytest.approx(2.0, rel=0.1)
