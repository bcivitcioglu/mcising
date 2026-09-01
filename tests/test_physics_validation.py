"""Physics validation tests using known analytical results.

These tests verify that the simulation produces physically correct results
for well-known cases of the 2D Ising model.
"""

from __future__ import annotations

import functools
import math
from typing import Final

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import Algorithm, ExecutionMode, LatticeConfig, SimulationConfig
from mcising.constants import TC_SQUARE_2D
from mcising.simulation import Simulation, SimulationResults

from tests._analytic import complete_elliptic_k, onsager_energy_per_site
from tests._stats import (
    DEFAULT_SEEDS,
    assert_mean_above,
    assert_mean_below,
    assert_ordered_means,
    assert_over_seeds,
    assert_samples_agree,
    assert_within_sigma,
    tau_int_blocking,
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
        sim.sweep(100, temperature=2.0)
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
        """At T=20 (beta=0.05), <E>/site = -2 tanh(beta) up to O(tanh^3).

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
        sim.sweep(100, temperature=20.0)
        energies = np.empty(200)
        for i in range(200):
            sim.sweep(10, temperature=20.0)
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
        sim.sweep(1000, temperature=2.0)

        energies = np.empty(100)
        for i in range(100):
            sim.sweep(5, temperature=2.0)
            energies[i] = sim.energy()

        assert_samples_agree(
            energies[:50],
            energies[50:],
            label_a=f"first half (seed={seed})",
            label_b=f"second half (seed={seed})",
        )


# ── True detailed-balance harness (P04) ──────────────────────────────
#
# A 3x3 square torus has only 512 spin states, so the full stationary
# distribution is checkable against exact Boltzmann weights computed by
# an in-test enumeration (independent of the Rust oracle).

DB_SIZE: Final = 3
DB_N_STATES: Final = 512
DB_TEMPERATURE: Final = 4.0
DB_N_SAMPLES: Final = 120_000
DB_INTERVAL: Final = 5
DB_THERMALIZATION: Final = 2_000
#: z=5 normal sigmas on the chi-square law of the G statistic
#: (one-test false-positive rate ~3e-7); see TestDetailedBalance.
DB_SIGMA: Final = 5.0


def _state_energies(size: int) -> np.ndarray:
    """Total energy of every spin state of a size x size torus at J1=+1.

    State s encodes site i = row*size + col as bit i (set bit = spin up),
    matching the row-major flat order of get_spins()/production_sweeps.
    Bonds are counted once via the +row and +col torus directions.
    """
    n = size * size
    states = np.arange(1 << n, dtype=np.int64)
    spins = np.where((states[:, None] >> np.arange(n)) & 1 == 1, 1, -1)
    energies = np.zeros(1 << n)
    for row in range(size):
        for col in range(size):
            site = row * size + col
            right = row * size + (col + 1) % size
            down = ((row + 1) % size) * size + col
            energies -= spins[:, site] * (spins[:, right] + spins[:, down])
    return energies


def _boltzmann(energies: np.ndarray, temperature: float) -> np.ndarray:
    weights = np.exp(-(energies - energies.min()) / temperature)
    return weights / weights.sum()


def _state_indices(configs: np.ndarray) -> np.ndarray:
    """Map (n, size, size) spin arrays to the integer state indices above."""
    n_samples = configs.shape[0]
    bits = (configs.reshape(n_samples, -1) > 0).astype(np.int64)
    return bits @ (1 << np.arange(bits.shape[1], dtype=np.int64))


def _kl_divergence(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """KL(empirical || exact); empty cells contribute 0 (0 log 0 = 0)."""
    q = counts / counts.sum()
    mask = q > 0
    return float(np.sum(q[mask] * np.log(q[mask] / probabilities[mask])))


def _chi2_upper(df: int, z: float) -> float:
    """Wilson-Hilferty upper quantile of chi-square_df at z normal sigmas.

    The naive normal approximation df + z*sqrt(2 df) is too tight for the
    right-skewed chi-square (measurably so at df~500); Wilson-Hilferty is
    accurate to <1% here.
    """
    return df * (1.0 - 2.0 / (9 * df) + z * math.sqrt(2.0 / (9 * df))) ** 3


def _kl_threshold(n_cells: int) -> float:
    return _chi2_upper(n_cells - 1, DB_SIGMA) / (2.0 * DB_N_SAMPLES)


def _sweep_support(size: int) -> np.ndarray:
    """Boolean successor matrix of one typewriter Metropolis sweep.

    Only the branching structure matters: a flip with dE <= 0 is accepted
    with probability exactly 1 (min(1, e^{-b dE}) = 1), so those rows are
    deterministic, while dE > 0 flips branch into {flip, stay}. The
    support — and hence the ergodic class structure — is therefore
    independent of temperature for 0 < T < inf.
    """
    n = size * size
    k = 1 << n
    energies = _state_energies(size)
    rows = np.arange(k)
    support = np.eye(k, dtype=bool)
    for site in range(n):
        flipped = rows ^ (1 << site)
        de = energies[flipped] - energies[rows]
        site_support = np.zeros((k, k), dtype=bool)
        site_support[rows, flipped] = True
        site_support[rows[de > 0], rows[de > 0]] = True
        # float32 matmul is BLAS-fast and exact for counts < 2^24.
        support = (support.astype(np.float32) @ site_support.astype(np.float32)) > 0
    return support


@functools.cache
def _metropolis_ergodic_class_mask() -> np.ndarray:
    """Mask of the 3x3 states reachable by typewriter Metropolis (#32).

    The sweep kernel is NOT irreducible: states where every flip is
    downhill are updated deterministically and form closed 2-cycles
    {s, global flip of s} — two frustrated checkerboards (E=+6) and three
    diagonal-stripe pairs (E=-2), 8 states in all, 0.98% of the Boltzmann
    mass at T=4. This computes the big class exactly from the sweep
    support digraph (transitive closure by repeated squaring).
    """
    support = _sweep_support(DB_SIZE)
    k = support.shape[0]
    reach = support | np.eye(k, dtype=bool)
    while True:
        closure = (reach.astype(np.float32) @ reach.astype(np.float32)) > 0
        closure |= reach
        if (closure == reach).all():
            break
        reach = closure
    mutual = reach & reach.T
    # The all-up ground state branches (every flip is uphill), so it lies
    # in the big class; take its communicating class.
    mask = mutual[k - 1]
    # The class must be closed (no support edge leaves it) and match the
    # exact decomposition documented in #32: 504 of 512 states.
    assert not support[mask][:, ~mask].any()
    assert int(mask.sum()) == 504
    return mask


class TestDetailedBalance:
    """The sampled state distribution is the exact Boltzmann distribution.

    TestStationarity above checks that <E> is stationary — necessary but
    far from sufficient (the B1 accept-everything bug was stationary too,
    just around the wrong distribution). Here every state of a 3x3 torus
    gets an exact Boltzmann weight and an empirical visit frequency.

    Scope: with sequential (typewriter) site updates the composite sweep
    satisfies balance, not per-move detailed balance; what is falsifiable
    from a trajectory — and what this test checks — is that the chain's
    stationary distribution is exactly Boltzmann on its ergodic class.

    Ergodicity caveat (#32, found BY this test): the typewriter Metropolis
    sweep kernel is reducible — 8 states where every flip is downhill form
    deterministic 2-cycles that never communicate with the other 504
    states (0.98% of Boltzmann mass at T=4). Within the big class the
    stationary measure is exactly the restricted Boltzmann distribution
    (verified to machine precision from the exact 512-state kernel), so
    Metropolis is compared against that restriction, with a canary
    assertion that the trapped states are never visited: when the scan-
    order fix for #32/#26 lands, the canary fails and this test must be
    unified back to the full distribution. Wolff and Swendsen-Wang are
    properly ergodic and face the full 512-state Boltzmann distribution.

    Threshold derivation: the G statistic against a fully specified
    distribution is G^2 = 2*N*KL(q_hat || p) over K cells; under the null
    it is asymptotically chi-square with df = K-1, so KL does NOT tend to
    zero: E[KL] = df/(2N) ~ 0.00213 at K = 512, N = 120,000. The
    threshold is the Wilson-Hilferty chi-square upper quantile at z=5
    (one-test tail ~3e-7), KL_max = chi2_{K-1}(z=5)/(2N) ~ 0.00286.
    Validity requires every expected count >~ 10: the rarest state at
    T=4.0 has p_min ~ 2.1e-4, giving expected count ~25 (asserted
    in-test). That is why T=4.0: at T=2.269 the same condition needs
    ~1.2e6 samples, at T=1.0 ~5e11.

    Power: the B1 failure mode (accept-everything -> near-uniform) gives
    KL ~ 0.7, ~250x the threshold; a sampler equilibrated at T=3.5
    instead of 4.0 gives ~10x the threshold (checked below without
    Monte Carlo); the #32 ergodicity defect itself showed up as 4.2x.
    """

    def _assert_visit_histogram_is_boltzmann(
        self, algorithm: str, seed: int
    ) -> None:
        exact_energies = _state_energies(DB_SIZE)
        p_full = _boltzmann(exact_energies, DB_TEMPERATURE)

        sim = IsingSimulation(DB_SIZE, 1.0, 0.0, 0.0, 0.0, seed, algorithm)
        sim.sweep(DB_THERMALIZATION, temperature=DB_TEMPERATURE)
        energies, mags, configs, _ = sim.production_sweeps(
            DB_N_SAMPLES, DB_INTERVAL, temperature=DB_TEMPERATURE, store_configs=True
        )
        idx = _state_indices(np.asarray(configs))

        # Free cross-check: the in-test energy table agrees with the Rust
        # per-site energies for every sampled state — validates the bit
        # mapping and the energy definition at once.
        np.testing.assert_allclose(
            exact_energies[idx] / (DB_SIZE * DB_SIZE), energies, atol=1e-12
        )

        # The threshold assumes ~independent samples. Convert residual
        # autocorrelation into a legible failure instead of a silent
        # threshold error. Fix: raise DB_INTERVAL, never the threshold.
        tau = max(
            tau_int_blocking(np.asarray(energies)),
            tau_int_blocking(np.abs(np.asarray(mags))),
        )
        assert tau < 1.0, f"samples correlated (tau_int={tau:.2f}); raise DB_INTERVAL"

        counts = np.bincount(idx, minlength=DB_N_STATES).astype(np.float64)

        if algorithm == "metropolis":
            mask = _metropolis_ergodic_class_mask()
            # Canary for #32: the trapped 2-cycles are unreachable today.
            # When the scan-order fix lands this fails — unify the test to
            # the full Boltzmann distribution then.
            assert counts[~mask].sum() == 0, (
                "typewriter Metropolis visited a state outside its ergodic "
                "class — #32 must be fixed; compare against the FULL "
                "Boltzmann distribution now"
            )
        else:
            mask = np.ones(DB_N_STATES, dtype=bool)

        p = p_full[mask] / p_full[mask].sum()
        # Chi-square validity of the threshold: asserted, not assumed.
        assert DB_N_SAMPLES * p.min() >= 10.0

        kl = _kl_divergence(counts[mask], p)
        n_cells = int(mask.sum())
        threshold = _kl_threshold(n_cells)
        null_mean = (n_cells - 1) / (2.0 * DB_N_SAMPLES)
        assert kl <= threshold, (
            f"{algorithm} (seed={seed}): KL(empirical||Boltzmann) = {kl:.5f} "
            f"> threshold {threshold:.5f} over {n_cells} states (null mean "
            f"{null_mean:.5f}, tau={tau:.2f}, min expected count "
            f"{DB_N_SAMPLES * p.min():.1f})"
        )

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_metropolis_visit_histogram_is_boltzmann(self, seed: int) -> None:
        self._assert_visit_histogram_is_boltzmann("metropolis", seed)

    @pytest.mark.statistical
    @pytest.mark.parametrize("algorithm", ["wolff", "swendsen_wang"])
    def test_cluster_visit_histogram_is_boltzmann(self, algorithm: str) -> None:
        self._assert_visit_histogram_is_boltzmann(algorithm, 42)

    def test_kl_threshold_resolves_a_wrong_temperature(self) -> None:
        """Power check without Monte Carlo: sampling at T=3.5 instead of
        4.0 (a ~12% temperature error) exceeds the threshold several-fold,
        so the gate resolves far subtler errors than the B1 bug class."""
        energies = _state_energies(DB_SIZE)
        p_target = _boltzmann(energies, DB_TEMPERATURE)
        p_wrong = _boltzmann(energies, 3.5)
        kl = float(np.sum(p_wrong * np.log(p_wrong / p_target)))
        assert kl > 5.0 * _kl_threshold(DB_N_STATES)


class TestFieldEffect:
    """External field h should bias magnetization in its direction."""

    @pytest.mark.statistical
    def test_positive_field_positive_magnetization(self) -> None:
        """h=+2 at T=1 aligns spins: <m> -> +1 (regime boundary 0.5)."""

        def check(seed: int) -> None:
            sim = IsingSimulation(8, 1.0, 0.0, 0.0, 2.0, seed)
            sim.sweep(500, temperature=1.0)
            mags = np.empty(50)
            for i in range(50):
                sim.sweep(5, temperature=1.0)
                mags[i] = sim.magnetization()
            assert_mean_above(mags, 0.5, label=f"<m>(h=+2, seed={seed})")

        assert_over_seeds(check)

    @pytest.mark.statistical
    def test_negative_field_negative_magnetization(self) -> None:
        """h=-2 at T=1 anti-aligns spins: <m> -> -1."""

        def check(seed: int) -> None:
            sim = IsingSimulation(8, 1.0, 0.0, 0.0, -2.0, seed)
            sim.sweep(500, temperature=1.0)
            mags = np.empty(50)
            for i in range(50):
                sim.sweep(5, temperature=1.0)
                mags[i] = sim.magnetization()
            assert_mean_below(mags, -0.5, label=f"<m>(h=-2, seed={seed})")

        assert_over_seeds(check)


# --- Onsager exact solution ----------------------------------------------

#: Lattice size for the Onsager comparisons. The finite-lattice energy at Tc
#: deviates from the thermodynamic limit by ~c/L with c ~ 0.62 (the exact 4x4
#: value -1.5658 gives 0.61; measured 1.36% at L=32 and 0.64% at L=64), so
#: L=32 fails a 1% gate on systematics alone and L=64 is the smallest power
#: of two that fits with room for the statistical error.
ONSAGER_L: Final = 64
#: Off-critical temperatures for the u(T) curve. Tc itself is excluded: the
#: exact formula is 0*inf there (its limit -sqrt(2) has its own test), and
#: away from Tc the finite-size correction is exp(-L/xi) < 1e-5 at L=64.
ONSAGER_CURVE_TEMPERATURES: Final = (1.5, 2.0, 2.5, 3.0, 3.5)


def _run_onsager(
    seed: int, temperatures: tuple[float, ...], n_sweeps: int, *, algorithm: Algorithm
) -> SimulationResults:
    config = SimulationConfig(
        lattice=LatticeConfig(size=ONSAGER_L),
        algorithm=algorithm,
        mode=ExecutionMode.INDEPENDENT,
        temperatures=temperatures,
        n_sweeps=n_sweeps,
        n_thermalization=2_000,
        measurement_interval=5,
        store_configs=False,  # N comes from the config; 8000 x 64x64 configs = 32 MB/T
        seed=seed,
    )
    return Simulation(config).run(show_progress=False)


class TestOnsagerEnergy:
    """Square-lattice internal energy against Onsager's exact solution.

    Two cluster algorithms, each where it decorrelates. Wolff at Tc: one
    cluster spans ~L^(7/4) ~ 35% of the 64x64 lattice, and the energy
    series has tau_int of 1-2 samples at 5 clusters per sample. Swendsen-
    Wang off-critical: in the disordered phase a single Wolff cluster
    flips only a few dozen spins, and at 5 clusters per sample the series
    stayed so correlated that the 8-block error could not resolve tau
    (calibration: pooled -3.2 sigma at T=3.0 with tau_int reported as
    ~40 samples, gone at 10x the thermalization) — the quoted error was
    not honest. SW updates every site each sweep and has tau_int < 1.2
    samples at every temperature here. Both run in independent mode:
    cluster updates equilibrate from a random start even below Tc (no
    quench trap).
    """

    def test_exact_helper_limits(self) -> None:
        """Pin the reference helpers before trusting them as oracles."""
        assert abs(complete_elliptic_k(0.0) - math.pi / 2.0) < 1e-15
        # K(1/sqrt(2)) = Gamma(1/4)^2 / (4 sqrt(pi)), a tabulated constant.
        assert abs(complete_elliptic_k(math.sqrt(0.5)) - 1.8540746773013719) < 1e-12
        for side in (1.0 - 1e-6, 1.0 + 1e-6):
            u = onsager_energy_per_site(TC_SQUARE_2D * side)
            assert abs(u + math.sqrt(2.0)) < 1e-4, (side, u)
        # High-temperature limit: u -> -(z/2) tanh(beta) + O(beta^3).
        assert abs(onsager_energy_per_site(100.0) + 2.0 * math.tanh(0.01)) < 1e-5
        curve = [onsager_energy_per_site(t) for t in ONSAGER_CURVE_TEMPERATURES]
        assert all(a < b for a, b in zip(curve, curve[1:])), curve

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_energy_at_tc_within_one_percent(self, seed: int) -> None:
        """<E>/site at Tc on 64x64 reproduces Onsager's -sqrt(2) within 1%.

        Tolerance budget: 0.64% finite-size systematic at L=64 (see
        ONSAGER_L) plus 4 sigma_stat, where sigma_stat <= 0.0025 (0.18%)
        is enforced as a canary so the run cannot be shortened until noise
        eats the margin. The blocking error of the mean is the quoted
        uncertainty. Calibration over DEFAULT_SEEDS (release build):
        deviations 0.64-0.77%, errors 0.0009-0.0012, worst margin to the
        1% line 2.6 sigma_stat; the 5-seed mean -1.4240(5) puts the L=64
        offset at 0.69% — the c/L systematic, not noise.
        """
        results = _run_onsager(
            seed, (TC_SQUARE_2D,), n_sweeps=40_000, algorithm=Algorithm.WOLFF
        )
        est = results.statistics(TC_SQUARE_2D).energy
        exact = -math.sqrt(2.0)
        deviation = abs(est.value - exact)
        assert est.error <= 0.0025, f"error budget exceeded: <E>/site = {est}"
        assert deviation <= 0.01 * abs(exact), (
            f"<E>/site(Tc, L={ONSAGER_L}, seed={seed}) = {est} vs Onsager "
            f"{exact:.6f}: |dev| = {deviation:.5f} "
            f"({100.0 * deviation / abs(exact):.2f}%) exceeds the 1% tolerance "
            f"({0.01 * abs(exact):.5f})"
        )

    @pytest.mark.slow
    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_internal_energy_curve_matches_onsager(self, seed: int) -> None:
        """<E>/site within 4 sigma of the exact u(T) at 5 off-critical temperatures.

        4000 samples per temperature (20000 Swendsen-Wang sweeps at
        interval 5; see the class docstring for why not Wolff). Two-sided
        against the exact infinite-lattice value: at L=64 the finite-size
        shift exp(-L/xi) is below 1e-5 at every temperature here, far
        below the ~1e-3 blocking errors, so no systematic allowance is
        needed. Calibration over DEFAULT_SEEDS (release build): worst
        1.80 sigma, every per-temperature pooled deviation within 1.3
        sigma, tau_int 0.5-1.2 samples, ~2 s per seed.
        """
        results = _run_onsager(
            seed,
            ONSAGER_CURVE_TEMPERATURES,
            n_sweeps=20_000,
            algorithm=Algorithm.SWENDSEN_WANG,
        )
        for t in ONSAGER_CURVE_TEMPERATURES:
            assert_within_sigma(
                results.energy[t],
                onsager_energy_per_site(t),
                label=f"<E>/site (T={t}, L={ONSAGER_L}, seed={seed})",
            )
