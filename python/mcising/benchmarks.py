"""Benchmark baselines for performance comparison.

Contains pure Python and NumPy-vectorized Metropolis implementations
for benchmarking against the Rust core. These are intentionally minimal
— just enough to measure the hot loop fairly.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Final

import numpy as np

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

    @property
    def total_updates(self) -> int:
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
    beta = 1.0 / 2.269
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
# NumPy-vectorized checkerboard Metropolis
# ---------------------------------------------------------------------------


def _numpy_metropolis(
    lattice_size: int, n_sweeps: int, beta: float, seed: int
) -> tuple[float, float, float]:
    """Run random single-spin-flip Metropolis using NumPy arrays.

    Same algorithm as pure Python (random site, compute dE, accept/reject),
    but uses NumPy arrays for spin storage and neighbor lookup tables.
    Returns (elapsed, energy, mag).
    """
    rng = np.random.default_rng(seed)
    n = lattice_size * lattice_size

    # Initialize random spins as flat NumPy array
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

    # Precompute neighbor table as (N, 4) array
    nn = np.empty((n, 4), dtype=np.intp)
    for idx in range(n):
        row, col = divmod(idx, lattice_size)
        nn[idx, 0] = ((row - 1) % lattice_size) * lattice_size + col
        nn[idx, 1] = ((row + 1) % lattice_size) * lattice_size + col
        nn[idx, 2] = row * lattice_size + (col - 1) % lattice_size
        nn[idx, 3] = row * lattice_size + (col + 1) % lattice_size

    # Precompute exp table for possible dE values
    exp_table = {de: math.exp(-beta * de) for de in (-8, -4, 0, 4, 8)}

    # Warmup
    for _ in range(100):
        for _ in range(n):
            idx = int(rng.integers(n))
            s = int(spins[idx])
            local_field = int(
                spins[nn[idx, 0]]
                + spins[nn[idx, 1]]
                + spins[nn[idx, 2]]
                + spins[nn[idx, 3]]
            )
            de = 2 * s * local_field
            if de <= 0 or rng.random() < exp_table[de]:
                spins[idx] = -spins[idx]

    # Timed run
    start = time.perf_counter()
    for _ in range(n_sweeps):
        for _ in range(n):
            idx = int(rng.integers(n))
            s = int(spins[idx])
            local_field = int(
                spins[nn[idx, 0]]
                + spins[nn[idx, 1]]
                + spins[nn[idx, 2]]
                + spins[nn[idx, 3]]
            )
            de = 2 * s * local_field
            if de <= 0 or rng.random() < exp_table[de]:
                spins[idx] = -spins[idx]
    elapsed = time.perf_counter() - start

    # Observables
    energy = 0.0
    for idx in range(n):
        s = int(spins[idx])
        row, col = divmod(idx, lattice_size)
        right = row * lattice_size + (col + 1) % lattice_size
        down = ((row + 1) % lattice_size) * lattice_size + col
        energy -= s * (int(spins[right]) + int(spins[down]))
    energy_per_site = energy / n

    mag_per_site = float(np.mean(spins))

    return elapsed, energy_per_site, mag_per_site


def bench_numpy(
    lattice_size: int, n_sweeps: int, seed: int = 42
) -> BenchmarkResult:
    """Benchmark NumPy-array Metropolis (same algorithm, NumPy storage)."""
    beta = 1.0 / 2.269
    elapsed, energy, mag = _numpy_metropolis(
        lattice_size, n_sweeps, beta, seed
    )
    return BenchmarkResult(
        name="NumPy",
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
) -> BenchmarkResult:
    """Benchmark mcising Rust core."""
    from mcising._core import IsingSimulation

    sim = IsingSimulation(lattice_size, 1.0, 0.0, 0.0, seed, algorithm)
    beta = 1.0 / 2.269

    # Warmup
    sim.sweep(100, beta)

    # Timed run
    start = time.perf_counter()
    sim.sweep(n_sweeps, beta)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=f"mcising ({algorithm})",
        lattice_size=lattice_size,
        n_sweeps=n_sweeps,
        elapsed=elapsed,
        energy=sim.energy(),
        magnetization=sim.magnetization(),
    )


# ---------------------------------------------------------------------------
# peapods (external Rust/PyO3 competitor)
# ---------------------------------------------------------------------------


def bench_peapods(
    lattice_size: int, n_sweeps: int, seed: int = 42
) -> BenchmarkResult:
    """Benchmark peapods Metropolis.

    Requires: ``uv sync --group benchmark``
    """
    import numpy as np
    from peapods import Ising  # type: ignore[import-untyped]

    model = Ising(
        (lattice_size, lattice_size),
        temperatures=np.array([2.269]),
        couplings="ferro",
    )

    # Set deterministic seed via internal core
    model._sim.reset(seed=seed)

    # Warmup
    model.sample(
        n_sweeps=100, sweep_mode="metropolis", warmup_ratio=0.0
    )

    # Timed run
    start = time.perf_counter()
    model.sample(
        n_sweeps=n_sweeps, sweep_mode="metropolis", warmup_ratio=0.0
    )
    elapsed = time.perf_counter() - start

    # peapods uses positive energy convention; negate for consistency
    energy = float(-model.energies_avg[0])
    magnetization = float(model.mags[0])

    return BenchmarkResult(
        name="peapods",
        lattice_size=lattice_size,
        n_sweeps=n_sweeps,
        elapsed=elapsed,
        energy=energy,
        magnetization=magnetization,
    )
