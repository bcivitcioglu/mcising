"""Antiferromagnetic (J<0) Metropolis correctness tests.

Regression suite for the energy-sign defect (B1): the single-coupling
sweep strategies branched on the neighbor-sum sign instead of the energy
sign, so for J<0 every proposed flip was accepted and the sampler
produced T=infinity configurations at every requested temperature.

The chain lattice is excluded from the equilibrium tests below: its
sequential-sweep dynamics stalls on constant-energy orbits and never
equilibrates at any temperature or coupling sign (issue #26, pre-existing
and independent of the B1 fix).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from mcising._core import IsingSimulation

from tests._stats import (
    DEFAULT_SEEDS,
    assert_mean_below,
    assert_over_seeds,
    assert_samples_agree,
    assert_within_sigma,
)

# Anneal ladder used to reach beta=2 without quench traps (see
# test_metropolis.py for the protocol rationale), extended to beta=10
# for the ground-state tests. Extra rungs near beta=0.22 and 0.44 pass
# slowly through the cubic and square/honeycomb transitions.
LADDER_TO_BETA2 = (0.2, 0.25, 0.3, 0.4, 0.44, 0.5, 0.7, 1.0, 1.5, 2.0)
LADDER_TO_BETA10 = LADDER_TO_BETA2 + (4.0, 10.0)

# (lattice_type, size): sizes are even so the Neel state is compatible
# with periodic boundaries.
ACCEPTANCE_LATTICES = [
    ("square", 8),
    ("triangular", 6),
    ("honeycomb", 6),
    ("cubic", 4),
]

# (lattice_type, size, E/site of the Neel state at J1=-1) — the Neel
# energy is -z_nn/2 * |J1|. Triangular is frustrated (no Neel state) and
# appears only in the acceptance test.
BIPARTITE = [
    ("square", 8, -2.0),
    ("honeycomb", 6, -1.5),
    ("cubic", 4, -3.0),
]


class TestAFMAcceptance:
    @pytest.mark.statistical
    @pytest.mark.parametrize(("lattice", "size"), ACCEPTANCE_LATTICES)
    def test_low_temp_acceptance_far_below_one(
        self, lattice: str, size: int
    ) -> None:
        """Pre-fix, J1<0 accepted every flip: the rate was exactly 1.0.

        Post-fix at beta=2 the bipartite lattices sit in the Neel phase
        where the cheapest flip costs dE = 2*z*|J1| (square: exp(-16) ~
        1e-7); measured rates are <0.001. The 0.35 bound is set by
        frustrated triangular, whose ground-state manifold keeps a finite
        fraction of zero-energy (always accepted) flips: measured
        equilibrium acceptance is 0.21-0.31 across the seed tuple, so
        0.35 clears the physical value with margin while rejecting the
        pre-fix 1.0 unambiguously.
        """

        def check(seed: int) -> None:
            sim = IsingSimulation(
                size, -1.0, 0.0, 0.0, 0.0, seed, "metropolis", lattice
            )
            for beta in LADDER_TO_BETA2:
                sim.sweep(200, temperature=1.0 / beta)
            rates = np.empty(200)
            for i in range(200):
                accepted, attempted, _ = sim.sweep(1, temperature=0.5)
                rates[i] = accepted / attempted
            assert_mean_below(
                rates, 0.35, label=f"{lattice} acceptance (seed={seed})"
            )

        assert_over_seeds(check)


class TestNeelGroundState:
    @pytest.mark.statistical
    @pytest.mark.parametrize(("lattice", "size", "e_neel"), BIPARTITE)
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_energy_reaches_neel_value(
        self, lattice: str, size: int, e_neel: float, seed: int
    ) -> None:
        """Annealed to beta=10 a bipartite AFM reaches the Neel state.

        E/site = -z_nn/2 * |J1| exactly. At beta=10 the cheapest
        excitation costs dE = 2*z_nn*|J1| (acceptance exp(-60) or less on
        these lattices), so the measured series must sit at the exact
        value and the 4-sigma comparison reduces to exact agreement.
        Calibration (release build): all 5 seeds land exactly on the
        Neel energy for all three lattices.
        """
        sim = IsingSimulation(size, -1.0, 0.0, 0.0, 0.0, seed, "metropolis", lattice)
        for beta in LADDER_TO_BETA10:
            sim.sweep(300, temperature=1.0 / beta)
        energies = np.empty(100)
        for i in range(100):
            sim.sweep(1, temperature=0.1)
            energies[i] = sim.energy()
        assert_within_sigma(
            energies, e_neel, label=f"{lattice} E/site (seed={seed})"
        )


class TestSinglePathMatchesMultiPath:
    """The fixed sign-branch sweeps against the always-safe branchless tables.

    Strategy dispatch keys on which couplings are nonzero, so the
    multi-coupling code path is forced with a second coupling of 1e-15:
    it selects the two-coupling table while perturbing dE by ~1e-15 |J| —
    far below statistical resolution. Runs at beta=0.35 (disordered side
    of every relevant transition) where equilibration is fast and energy
    fluctuations give an honest blocking error. Pre-fix, the J<0
    single-coupling paths sampled T=infinity (E ~ 0) while the
    multi-coupling paths sampled correctly — a disagreement of ~0.5,
    orders of magnitude beyond 4 sigma.
    """

    EPS = 1e-15

    @staticmethod
    def _energies(j1: float, j2: float, j3: float, seed: int) -> np.ndarray:
        sim = IsingSimulation(8, j1, j2, j3, 0.0, seed)
        for beta in (0.2, 0.3, 0.35):
            sim.sweep(200, temperature=1.0 / beta)
        energies = np.empty(1500)
        for i in range(1500):
            sim.sweep(1, temperature=1.0 / 0.35)
            energies[i] = sim.energy()
        return energies

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_j1_negative(self, seed: int) -> None:
        single = self._energies(-1.0, 0.0, 0.0, seed)
        multi = self._energies(-1.0, self.EPS, 0.0, seed + 1000)
        assert_samples_agree(
            single, multi, label_a="sweep_j1 (J1=-1)", label_b="sweep_j1j2 (J2=eps)"
        )

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_j2_negative(self, seed: int) -> None:
        single = self._energies(0.0, -1.0, 0.0, seed)
        multi = self._energies(self.EPS, -1.0, 0.0, seed + 1000)
        assert_samples_agree(
            single, multi, label_a="sweep_j2 (J2=-1)", label_b="sweep_j1j2 (J1=eps)"
        )

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_j3_negative(self, seed: int) -> None:
        single = self._energies(0.0, 0.0, -1.0, seed)
        multi = self._energies(self.EPS, 0.0, -1.0, seed + 1000)
        assert_samples_agree(
            single, multi, label_a="sweep_j3 (J3=-1)", label_b="sweep_j1j3 (J1=eps)"
        )


# --- exact ground states (static, no dynamics) ---------------------------


def _square_neel(size: int) -> np.ndarray:
    rows, cols = np.indices((size, size))
    return np.where((rows + cols) % 2 == 0, 1, -1).astype(np.int8)


def _chain_neel(size: int) -> np.ndarray:
    return np.where(np.arange(size) % 2 == 0, 1, -1).astype(np.int8)


def _honeycomb_neel(size: int) -> np.ndarray:
    # Spins are indexed (row, col, sublattice) and every NN bond joins the
    # two sublattices, so a sublattice sign split is the Neel state.
    spins = np.empty((size, size, 2), dtype=np.int8)
    spins[:, :, 0] = 1
    spins[:, :, 1] = -1
    return spins


def _cubic_neel(size: int) -> np.ndarray:
    i, j, k = np.indices((size, size, size))
    return np.where((i + j + k) % 2 == 0, 1, -1).astype(np.int8)


def _triangular_stripe(size: int) -> np.ndarray:
    _, cols = np.indices((size, size))
    return np.where(cols % 2 == 0, 1, -1).astype(np.int8)


# (lattice_type, size, ground-state builder, exact E/site at J1=-1). Sizes
# are even so the patterns wrap consistently under periodic boundaries
# (triangular/honeycomb require even L anyway).
EXACT_GROUND_STATES: list[tuple[str, int, Callable[[int], np.ndarray], float]] = [
    ("square", 6, _square_neel, -2.0),
    ("chain", 10, _chain_neel, -1.0),
    ("honeycomb", 6, _honeycomb_neel, -1.5),
    ("cubic", 4, _cubic_neel, -3.0),
    ("triangular", 6, _triangular_stripe, -1.0),
]


class TestExactGroundStates:
    """First-principles AFM ground-state energies, checked statically.

    Extends the all-up grid of test_cross_lattice.py (FM ground state,
    E/site = -(J1 z_nn + J2 z_nnn + J3 z_tnn)/2 - h) to J1 = -1, where the
    ground state is no longer the trivial one. Configurations are set
    directly and only the energy is read, so no dynamics is involved and
    the chain is included despite #26.
    """

    @pytest.mark.parametrize(
        ("lattice", "size", "build", "e_exact"),
        EXACT_GROUND_STATES,
        ids=[row[0] for row in EXACT_GROUND_STATES],
    )
    def test_ground_state_energy(
        self,
        lattice: str,
        size: int,
        build: Callable[[int], np.ndarray],
        e_exact: float,
    ) -> None:
        """The constructed state has exactly the derived ground-state energy.

        Bipartite lattices (square, chain, honeycomb, cubic): the Neel
        state anti-aligns every NN bond, so all N z_nn / 2 bonds sit at
        their minimum -|J1| and E/site = -z_nn |J1| / 2. That is the global
        minimum because no bond can contribute less.

        Triangular (frustrated, no Neel state): three mutually anti-aligned
        spins are impossible, so every elementary triangle carries at
        least one frustrated (aligned, +|J1|) bond. With 2N triangles, 3N
        bonds and each bond shared by two triangles, at least N bonds are
        frustrated, giving E/site >= (-(3N - N) + N) |J1| / N = -|J1|. The
        column stripe frustrates exactly the two in-column bonds per site
        (N bonds total) and attains the bound, so it is a true ground state.
        """
        sim = IsingSimulation(size, -1.0, 0.0, 0.0, 0.0, 42, "metropolis", lattice)
        sim.set_spins(build(size))
        assert sim.magnetization() == 0.0, "pattern sanity: zero net moment"
        assert abs(sim.energy() - e_exact) < 1e-10, (
            f"{lattice} L={size} ground state: E/site={sim.energy():.12f} "
            f"expected {e_exact}"
        )
