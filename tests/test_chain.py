"""Tests for the 1D chain lattice."""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.simulation import Simulation, SimulationResults
from mcising.statistics import jackknife_estimate, susceptibility

from tests._analytic import chain_energy_per_site, chain_susceptibility_signed
from tests._stats import (
    DEFAULT_SEEDS,
    assert_estimate_within_sigma,
    assert_within_sigma,
)

#: Ring size and temperatures for the transfer-matrix comparison. N=64 keeps
#: the finite-N term t^N resolvable at T=0.8 (tanh(1.25)^64 ~ 3e-5) while the
#: correlation length xi = -1/ln(t) ~ 6 sites stays well inside the ring.
CHAIN_N = 64
CHAIN_TEMPERATURES = (0.8, 1.2, 1.8, 2.5, 3.5)


class TestChainEnergy:
    """Test energy computation on chain lattice."""

    def test_all_up_energy(self) -> None:
        """All spins up: E = -J1 * 2 / 2 = -1.0 per site."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        sim.set_spins(np.ones(10, dtype=np.int8))
        assert abs(sim.energy() - (-1.0)) < 1e-10

    def test_all_up_with_field(self) -> None:
        """All up, J1=1, h=1: E = -1.0 - 1.0 = -2.0 per site."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 1.0, 42, "metropolis", "chain")
        sim.set_spins(np.ones(10, dtype=np.int8))
        assert abs(sim.energy() - (-2.0)) < 1e-10

    def test_all_down_energy(self) -> None:
        """All down: same energy as all up."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        sim.set_spins(-np.ones(10, dtype=np.int8))
        assert abs(sim.energy() - (-1.0)) < 1e-10

    def test_alternating_energy(self) -> None:
        """Alternating +1/-1: every NN pair antiparallel, E = +1.0 per site."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        spins = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
        sim.set_spins(spins)
        assert abs(sim.energy() - 1.0) < 1e-10


class TestChainSimulation:
    """Test simulation behavior on chain lattice."""

    def test_energy_decreases_at_low_t(self) -> None:
        sim = IsingSimulation(50, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        e_before = sim.energy()
        sim.sweep(200, temperature=0.1)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_deterministic(self) -> None:
        sim1 = IsingSimulation(50, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "chain")
        sim2 = IsingSimulation(50, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "chain")
        sim1.sweep(10, temperature=2.0)
        sim2.sweep(10, temperature=2.0)
        np.testing.assert_array_equal(sim1.get_spins(), sim2.get_spins())

    def test_spins_shape_is_1d(self) -> None:
        """Chain should return a 1D spin array."""
        sim = IsingSimulation(20, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        spins = sim.get_spins()
        assert spins.ndim == 1
        assert spins.shape == (20,)


class TestChainCluster:
    """Test cluster algorithms on chain lattice."""

    def test_wolff_runs(self) -> None:
        sim = IsingSimulation(20, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "chain")
        accepted, attempted, _ = sim.sweep(10, temperature=2.0)
        assert attempted > 0

    def test_swendsen_wang_runs(self) -> None:
        sim = IsingSimulation(20, 1.0, 0.0, 0.0, 0.0, 42, "swendsen_wang", "chain")
        accepted, attempted, _ = sim.sweep(10, temperature=2.0)
        assert attempted > 0


class TestChainHighLevel:
    """Test high-level Simulation class with chain lattice."""

    def test_run_chain(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.CHAIN,
                size=50,
                j1=1.0,
            ),
            temperatures=(2.0, 1.0, 0.5),
            n_sweeps=50,
            measurement_interval=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 3


class TestChainPhysics:
    """Test 1D chain physics: no ordering at T > 0."""

    def test_no_ordering_at_finite_t(self) -> None:
        """1D Ising has Tc=0: at T=1.0, |m| should be small for large L."""
        sim = IsingSimulation(100, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        # Thermalize
        sim.sweep(1000, temperature=1.0)
        # Measure
        mags = []
        for _ in range(100):
            sim.sweep(10, temperature=1.0)
            mags.append(abs(sim.magnetization()))
        avg_mag = np.mean(mags)
        assert avg_mag < 0.5, (
            f"1D chain should not order at T=1.0, got <|m|>={avg_mag:.3f}"
        )


def _chain_trace(n: int, temperature: float) -> tuple[float, float]:
    """Brute-force <E>/site and signed chi over all 2^n states of the ring."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    spins = np.where(bits == 1, 1.0, -1.0)
    energies = -np.sum(spins * np.roll(spins, -1, axis=1), axis=1)
    weights = np.exp(-(energies - energies.min()) / temperature)
    z = float(weights.sum())
    e_site = float((weights * energies).sum()) / z / n
    m = spins.sum(axis=1) / n
    chi = n * float((weights * m * m).sum()) / z / temperature
    return e_site, chi


@functools.cache
def _run_chain_wolff(seed: int) -> SimulationResults:
    config = SimulationConfig(
        lattice=LatticeConfig(lattice_type=LatticeType.CHAIN, size=CHAIN_N),
        algorithm=Algorithm.WOLFF,
        mode=ExecutionMode.INDEPENDENT,
        temperatures=CHAIN_TEMPERATURES,
        n_sweeps=20_000,
        n_thermalization=2_000,
        measurement_interval=5,
        seed=seed,
    )
    return Simulation(config).run(show_progress=False)


def _signed_chi(temperature: float) -> Callable[[np.ndarray], float]:
    def estimator(m: np.ndarray) -> float:
        return susceptibility(
            m, temperature=temperature, num_sites=CHAIN_N, kind="signed"
        )

    return estimator


class TestChainClosedForms:
    """The 1D closed forms (Ising 1925) against Wolff runs on a 64-site ring.

    Wolff rather than Metropolis: sequential-sweep Metropolis never
    equilibrates the chain at any temperature (#26, open; its fix changes
    RNG streams and is a phase of its own). Cluster updates are ergodic
    here, and the Rust exact-enumeration oracle already pins chain-12
    Wolff to the transfer matrix within 0.5%; these tests do it at 4 sigma
    through the public API, with the finite-N formulas so no
    thermodynamic-limit hand-waving enters.
    """

    def test_closed_forms_match_enumeration(self) -> None:
        """Independent oracle for tests._analytic: a full 2^12-state trace."""
        for t in CHAIN_TEMPERATURES:
            e_trace, chi_trace = _chain_trace(12, t)
            assert abs(e_trace - chain_energy_per_site(12, t)) < 1e-10, t
            assert abs(chi_trace - chain_susceptibility_signed(12, t)) < 1e-10, t

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_energy_matches_transfer_matrix(self, seed: int) -> None:
        """<E>/site within 4 sigma of -(t + t^(N-1))/(1 + t^N), 5 temperatures.

        4000 samples per temperature (20000 Wolff sweeps at interval 5)
        give blocking errors of ~1e-3 with tau_int of 1-4 samples, so the
        two-sided comparison is honest (deviation ~ |N(0,1)| under the
        null). Calibration over DEFAULT_SEEDS (release build): worst 2.16
        sigma, pooled per-temperature deviations within 1.3 sigma.
        """
        results = _run_chain_wolff(seed)
        for t in CHAIN_TEMPERATURES:
            assert_within_sigma(
                results.energy[t],
                chain_energy_per_site(CHAIN_N, t),
                label=f"chain <E>/site (T={t}, seed={seed})",
            )

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_susceptibility_matches_transfer_matrix(self, seed: int) -> None:
        """Signed chi within 4 sigma (jackknife) of the closed form, 5 temperatures.

        The signed convention N Var(m)/T is compared because <m> = 0 in the
        full trace and N <m^2> has a closed form; the package's default
        "connected" N Var(|m|)/T does not. ``results.statistics(T)`` only
        quotes the connected jackknife, so the signed one is built here
        from the same delete-one-block machinery. Calibration over
        DEFAULT_SEEDS (release build): worst 1.70 sigma.
        """
        results = _run_chain_wolff(seed)
        for t in CHAIN_TEMPERATURES:
            est = jackknife_estimate(results.magnetization[t], _signed_chi(t))
            assert_estimate_within_sigma(
                est,
                chain_susceptibility_signed(CHAIN_N, t),
                label=f"chain chi_signed (T={t}, seed={seed})",
            )
