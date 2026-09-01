"""Benchmark baselines for performance comparison.

Contains a pure Python single-spin-flip loop and a NumPy checkerboard
(whole-array sublattice) Metropolis implementation for benchmarking
against the Rust core, plus runners for the Rust core itself and for the
external peapods package. These are intentionally minimal — just enough
to measure the hot loop fairly. ``benchmarks/run_all.py`` drives them to
regenerate every published number.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from mcising.constants import TC_SQUARE_2D

__all__: Final[list[str]] = [
    "BenchmarkResult",
    "bench_pure_python",
    "bench_numpy",
    "bench_mcising",
    "bench_peapods",
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    lattice_size: int
    n_sweeps: int
    elapsed: float
    energy: float
    magnetization: float
    num_sites: int | None = None
    attempted_updates: int | None = None

    @property
    def total_updates(self) -> int:
        # Real attempted-flip count when the backend reports one — for
        # Wolff, n_sweeps * num_sites would overstate the work by ~N/|C|
        # (one sweep is one cluster).
        if self.attempted_updates is not None:
            return self.attempted_updates
        if self.num_sites is not None:
            return self.n_sweeps * self.num_sites
        return self.n_sweeps * self.lattice_size * self.lattice_size

    @property
    def updates_per_sec(self) -> float:
        return self.total_updates / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def sweeps_per_sec(self) -> float:
        return self.n_sweeps / self.elapsed if self.elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Pure Python Metropolis (no NumPy in the hot loop)
# ---------------------------------------------------------------------------


def _pure_python_metropolis(
    lattice_size: int, n_sweeps: int, beta: float, seed: int
) -> tuple[float, float, float]:
    """Run Metropolis sweeps using pure Python. Returns (elapsed, energy, mag)."""
    rng = random.Random(seed)
    n = lattice_size * lattice_size

    # Initialize random spins as flat list
    spins: list[int] = [1 if rng.random() < 0.5 else -1 for _ in range(n)]

    # Precompute neighbor indices (periodic square lattice, nearest-neighbor only)
    nn: list[list[int]] = []
    for idx in range(n):
        row, col = divmod(idx, lattice_size)
        neighbors = [
            ((row - 1) % lattice_size) * lattice_size + col,
            ((row + 1) % lattice_size) * lattice_size + col,
            row * lattice_size + (col - 1) % lattice_size,
            row * lattice_size + (col + 1) % lattice_size,
        ]
        nn.append(neighbors)

    # Precompute exp table for possible dE values: dE = 2*s*sum_nn
    # sum_nn in {-4, -2, 0, 2, 4}, s in {-1, 1}, so dE in {-8, -4, 0, 4, 8}
    exp_table: dict[int, float] = {}
    for de in (-8, -4, 0, 4, 8):
        exp_table[de] = math.exp(-beta * de)

    # Warmup (100 sweeps)
    for _ in range(100):
        for _ in range(n):
            idx = rng.randrange(n)
            s = spins[idx]
            local_field = sum(spins[j] for j in nn[idx])
            de = 2 * s * local_field
            if de <= 0 or rng.random() < exp_table[de]:
                spins[idx] = -s

    # Timed run
    start = time.perf_counter()
    for _ in range(n_sweeps):
        for _ in range(n):
            idx = rng.randrange(n)
            s = spins[idx]
            local_field = sum(spins[j] for j in nn[idx])
            de = 2 * s * local_field
            if de <= 0 or rng.random() < exp_table[de]:
                spins[idx] = -s
    elapsed = time.perf_counter() - start

    # Observables
    energy = 0.0
    for idx in range(n):
        s = spins[idx]
        # Only count right and down neighbors to avoid double-counting
        row, col = divmod(idx, lattice_size)
        right = row * lattice_size + (col + 1) % lattice_size
        down = ((row + 1) % lattice_size) * lattice_size + col
        energy -= s * (spins[right] + spins[down])
    energy_per_site = energy / n

    mag_per_site = sum(spins) / n

    return elapsed, energy_per_site, mag_per_site


def bench_pure_python(
    lattice_size: int, n_sweeps: int, seed: int = 42
) -> BenchmarkResult:
    """Benchmark pure Python Metropolis."""
    beta = 1.0 / TC_SQUARE_2D
    elapsed, energy, mag = _pure_python_metropolis(lattice_size, n_sweeps, beta, seed)
    return BenchmarkResult(
        name="Pure Python",
        lattice_size=lattice_size,
        n_sweeps=n_sweeps,
        elapsed=elapsed,
        energy=energy,
        magnetization=mag,
    )


# ---------------------------------------------------------------------------
# NumPy checkerboard Metropolis (whole-array sublattice updates)
# ---------------------------------------------------------------------------


def _numpy_metropolis(
    lattice_size: int, n_sweeps: int, beta: float, seed: int
) -> tuple[float, float, float]:
    """Run checkerboard Metropolis with whole-array NumPy updates.

    The square lattice splits into two sublattices (sites with even and
    odd ``row + col``) that share no bonds, so every site of one
    sublattice can be updated at once: the local field comes from four
    rolled copies of the spin array, the Boltzmann factor is evaluated on
    the whole array and the accepted flips are applied through a mask.
    One sweep updates both sublattices, i.e. attempts every site once —
    the same work as a sequential single-spin-flip sweep, in a different
    order. Returns (elapsed, energy, mag).
    """
    if lattice_size % 2:
        raise ValueError(
            "checkerboard Metropolis needs an even lattice size so the two "
            f"periodic sublattices close, got {lattice_size}"
        )
    rng = np.random.default_rng(seed)
    shape = (lattice_size, lattice_size)
    spins: NDArray[np.int8] = rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)
    rows, cols = np.indices(shape)
    masks: list[NDArray[np.bool_]] = [(rows + cols) % 2 == parity for parity in (0, 1)]

    def sweep() -> None:
        nonlocal spins
        for mask in masks:
            field = (
                np.roll(spins, 1, axis=0)
                + np.roll(spins, -1, axis=0)
                + np.roll(spins, 1, axis=1)
                + np.roll(spins, -1, axis=1)
            )
            # dE = 2 s h with h in {-4, ..., 4}: stays within int8.
            de = 2 * spins * field
            accept = mask & ((de <= 0) | (rng.random(shape) < np.exp(-beta * de)))
            spins = np.where(accept, -spins, spins)

    # Warmup
    for _ in range(100):
        sweep()

    # Timed run
    start = time.perf_counter()
    for _ in range(n_sweeps):
        sweep()
    elapsed = time.perf_counter() - start

    # Observables: right and down neighbors only, so each bond counts once.
    bonds = np.roll(spins, -1, axis=0) + np.roll(spins, -1, axis=1)
    energy_per_site = -float(np.sum(spins * bonds, dtype=np.int64)) / spins.size
    mag_per_site = float(np.mean(spins))

    return elapsed, energy_per_site, mag_per_site


def bench_numpy(lattice_size: int, n_sweeps: int, seed: int = 42) -> BenchmarkResult:
    """Benchmark the NumPy checkerboard Metropolis (even ``lattice_size`` only)."""
    beta = 1.0 / TC_SQUARE_2D
    elapsed, energy, mag = _numpy_metropolis(lattice_size, n_sweeps, beta, seed)
    return BenchmarkResult(
        name="NumPy (checkerboard)",
        lattice_size=lattice_size,
        n_sweeps=n_sweeps,
        elapsed=elapsed,
        energy=energy,
        magnetization=mag,
    )


# ---------------------------------------------------------------------------
# mcising Rust core
# ---------------------------------------------------------------------------


def bench_mcising(
    lattice_size: int,
    n_sweeps: int,
    seed: int = 42,
    algorithm: str = "metropolis",
    lattice_type: str = "square",
    temperature: float = TC_SQUARE_2D,
) -> BenchmarkResult:
    """Benchmark mcising Rust core."""
    from mcising._core import IsingSimulation

    sim = IsingSimulation(
        lattice_size, 1.0, 0.0, 0.0, 0.0, seed, algorithm, lattice_type
    )
    num_sites = sim.num_sites

    # Warmup
    sim.sweep(100, temperature=temperature)

    # Timed run — sweeps only, observables computed once at the end
    start = time.perf_counter()
    _accepted, attempted, _cluster_flips = sim.sweep(n_sweeps, temperature=temperature)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=f"mcising ({algorithm})",
        lattice_size=lattice_size,
        n_sweeps=n_sweeps,
        elapsed=elapsed,
        energy=sim.energy(),
        magnetization=sim.magnetization(),
        num_sites=num_sites,
        attempted_updates=attempted,
    )


# ---------------------------------------------------------------------------
# peapods (external Rust/PyO3 competitor)
# ---------------------------------------------------------------------------


def bench_peapods(
    lattice_size: int,
    n_sweeps: int,
    seed: int = 42,
    *,
    geometry: str = "square",
    dim: int = 2,
    temperature: float = TC_SQUARE_2D,
    cluster_mode: str | None = None,
) -> BenchmarkResult:
    """Benchmark the external peapods package (one parametrized runner).

    Replaces the pre-1.0 ``bench_peapods_{triangular,cubic,wolff,sw}``
    near-copies. peapods is not a dependency of mcising: install it with
    ``uv sync --group benchmark``. The published comparison is produced
    by ``benchmarks/run_all.py``, whose matched-physics runners record
    the energy every sweep on both sides.
    """
    from peapods import Ising

    shape = (lattice_size,) * dim
    ctor_kwargs: dict[str, Any] = {"couplings": "ferro"}
    if geometry != "square":
        ctor_kwargs["geometry"] = geometry
    model = Ising(shape, temperatures=np.array([temperature]), **ctor_kwargs)
    model._sim.reset(seed=seed)

    sample_kwargs: dict[str, Any] = {
        "sweep_mode": "metropolis",
        "warmup_ratio": 0.0,
    }
    if cluster_mode is not None:
        sample_kwargs["cluster_update_interval"] = 1
        sample_kwargs["cluster_mode"] = cluster_mode

    # Warmup
    model.sample(n_sweeps=100, **sample_kwargs)

    # Timed run
    start = time.perf_counter()
    model.sample(n_sweeps=n_sweeps, **sample_kwargs)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="peapods",
        lattice_size=lattice_size,
        n_sweeps=n_sweeps,
        elapsed=elapsed,
        energy=float(-model.energies_avg[0]),
        magnetization=float(model.mags[0]),
        num_sites=lattice_size**dim,
    )
