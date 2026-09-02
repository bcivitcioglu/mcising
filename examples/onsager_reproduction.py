#!/usr/bin/env python3
"""Reproduce Onsager's exact solution of the square-lattice Ising model.

The internal energy per site u(T) of the infinite square lattice is known
in closed form (Onsager 1944) and the spontaneous magnetization m(T) below
Tc as well (Yang 1952):

    u(T) = -coth(2 beta) [1 + (2/pi) (2 tanh^2(2 beta) - 1) K(k1)],
    k1   = 2 sinh(2 beta) / cosh^2(2 beta),
    m(T) = [1 - sinh^-4(2 beta)]^(1/8)   for T < Tc, 0 above,

with beta = 1/T, J = 1, k_B = 1 and K the complete elliptic integral of the
first kind. This script measures both on periodic L x L lattices with the
Swendsen-Wang algorithm and draws them over the exact curves with the
blocking error bars quoted by ``SimulationResults.statistics``.

What to expect: wherever the lattice is much larger than the exact
correlation length xi(T) of the infinite system (L / xi >~ 8) every energy
point agrees with the exact curve within its error bar. Closer to Tc the
periodic lattice deviates systematically, by an amount that shrinks with
L / xi (exactly at Tc it sits about 0.6/L below -sqrt(2)) -- that is
finite-size rounding, not disagreement, and the table prints L / xi next
to every point so the two regimes can be told apart. <|m|> matches Yang's
curve deep in the ordered phase and rounds off the singularity at Tc with
a finite-size tail above it, which shrinks from L = 16 to L = 64.

The correlation length used for that column is Onsager's exact result,
xi^-1 = |2 beta + ln tanh beta| above Tc and twice that below (the
amplitude ratio xi+/xi- = 2 of the 2D Ising model), which vanishes
linearly at Tc with amplitude xi ~ 0.567 / |T/Tc - 1|.

Swendsen-Wang rather than Wolff: off-critical a single Wolff cluster flips
only a few dozen spins, so a fixed cluster interval leaves the series so
correlated that the quoted error is not honest (see the P14 physics-
validation tests).

Usage:
    python examples/onsager_reproduction.py                 # full budget
    python examples/onsager_reproduction.py --out figures/  # choose the directory
    python examples/onsager_reproduction.py --quick         # seconds, smoke test

Runtime (Apple M4, release build): see RUNTIME_NOTE below.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from mcising import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    Simulation,
    SimulationConfig,
    SimulationResults,
)
from mcising.constants import TC_SQUARE_2D
from numpy.typing import NDArray

RUNTIME_NOTE: Final = "measured wall time is printed at the end of the run"

FULL_SIZES: Final[tuple[int, ...]] = (16, 64)
FULL_TEMPERATURES: Final[tuple[float, ...]] = (
    1.5,
    1.7,
    1.9,
    2.0,
    2.1,
    2.2,
    2.25,
    TC_SQUARE_2D,
    2.3,
    2.35,
    2.45,
    2.6,
    2.8,
    3.0,
    3.5,
)
QUICK_SIZES: Final[tuple[int, ...]] = (8,)
QUICK_TEMPERATURES: Final[tuple[float, ...]] = (1.5, 2.0, TC_SQUARE_2D, 2.6, 3.5)


def complete_elliptic_k(k: float) -> float:
    """Complete elliptic integral of the first kind via the AGM.

    K(k) = pi / (2 AGM(1, sqrt(1 - k^2))); the iteration converges
    quadratically. Requires 0 <= k < 1 (K diverges logarithmically at 1).
    """
    if not 0.0 <= k < 1.0:
        raise ValueError(f"K(k) requires 0 <= k < 1, got {k}")
    a, b = 1.0, math.sqrt(1.0 - k * k)
    for _ in range(64):
        if abs(a - b) <= 1e-15 * a:
            break
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


def onsager_energy(temperature: float) -> float:
    """Exact internal energy per site of the infinite square lattice.

    At Tc the elliptic modulus is exactly 1 and the closed form is a
    0 * infinity limit whose value is -sqrt(2); that constant is returned.
    """
    if math.isclose(temperature, TC_SQUARE_2D, rel_tol=1e-12):
        return -math.sqrt(2.0)
    two_beta = 2.0 / temperature
    sinh, cosh = math.sinh(two_beta), math.cosh(two_beta)
    k1 = 2.0 * sinh / (cosh * cosh)
    tanh_sq = (sinh / cosh) ** 2
    bracket = 1.0 + (2.0 / math.pi) * (2.0 * tanh_sq - 1.0) * complete_elliptic_k(k1)
    return -(cosh / sinh) * bracket


def correlation_length(temperature: float) -> float:
    """Exact correlation length of the infinite square lattice (Onsager).

    xi^-1 = |2 beta + ln tanh beta| for T > Tc; below Tc the inverse length
    is twice that (amplitude ratio 2). Infinite exactly at Tc.
    """
    beta = 1.0 / temperature
    inverse = abs(2.0 * beta + math.log(math.tanh(beta)))
    if temperature < TC_SQUARE_2D:
        inverse *= 2.0
    return math.inf if inverse == 0.0 else 1.0 / inverse


def yang_magnetization(temperature: float) -> float:
    """Exact spontaneous magnetization of the infinite square lattice."""
    if temperature >= TC_SQUARE_2D:
        return 0.0
    return float((1.0 - math.sinh(2.0 / temperature) ** -4) ** 0.125)


def run(
    size: int,
    temperatures: Sequence[float],
    *,
    n_sweeps: int,
    n_thermalization: int,
    seed: int,
) -> SimulationResults:
    """Swendsen-Wang in independent mode: every temperature is its own chain."""
    config = SimulationConfig(
        lattice=LatticeConfig(size=size, j1=1.0),
        algorithm=Algorithm.SWENDSEN_WANG,
        mode=ExecutionMode.INDEPENDENT,
        temperatures=tuple(temperatures),
        n_sweeps=n_sweeps,
        n_thermalization=n_thermalization,
        measurement_interval=5,
        store_configs=False,
        seed=seed,
    )
    return Simulation(config).run(show_progress=False)


def exact_curves(
    t_min: float, t_max: float, n: int = 400
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Dense exact u(T) and m(T) for drawing, with Tc excluded from the grid."""
    grid = np.linspace(t_min, t_max, n)
    grid = grid[~np.isclose(grid, TC_SQUARE_2D)]
    energy = np.array([onsager_energy(float(t)) for t in grid])
    magnetization = np.array([yang_magnetization(float(t)) for t in grid])
    return grid, energy, magnetization


def make_figure(
    results: dict[int, SimulationResults], output: Path, *, budget: str
) -> Path:
    """Draw energy and magnetization against the exact curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    all_temps = sorted({t for r in results.values() for t in r.temperatures})
    grid, u_exact, m_exact = exact_curves(min(all_temps) - 0.1, max(all_temps) + 0.1)
    ax_e.plot(
        grid, u_exact, color="black", lw=1.2, label="Onsager (exact, $L\\to\\infty$)"
    )
    ax_m.plot(
        grid, m_exact, color="black", lw=1.2, label="Yang (exact, $L\\to\\infty$)"
    )

    markers = ("s", "o", "^", "D")
    for (size, res), marker in zip(sorted(results.items()), markers):
        temps = np.array(res.temperatures)
        stats = [res.statistics(t) for t in res.temperatures]
        ax_e.errorbar(
            temps,
            [s.energy.value for s in stats],
            yerr=[s.energy.error for s in stats],
            fmt=marker,
            ms=4,
            capsize=2,
            lw=1,
            label=f"$L={size}$",
        )
        ax_m.errorbar(
            temps,
            [s.abs_magnetization.value for s in stats],
            yerr=[s.abs_magnetization.error for s in stats],
            fmt=marker,
            ms=4,
            capsize=2,
            lw=1,
            label=f"$L={size}$",
        )

    for ax in (ax_e, ax_m):
        ax.axvline(TC_SQUARE_2D, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("$T$  ($J = k_B = 1$)")
        ax.legend(frameon=False, fontsize=8)
    ax_e.set_ylabel("energy per site $\\langle E \\rangle / N$")
    ax_m.set_ylabel("$\\langle |m| \\rangle$")
    ax_e.set_title("Internal energy", fontsize=10)
    ax_m.set_title("Magnetization", fontsize=10)
    fig.suptitle(
        f"Square-lattice Ising model, Swendsen-Wang, periodic $L\\times L$ ({budget})",
        fontsize=10,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


LARGE_LATTICE_RATIO: Final = (
    8.0  # L / xi above which finite-size effects are negligible
)


def report(results: dict[int, SimulationResults]) -> float:
    """Print a per-temperature table; return the worst |dev|/sigma at L/xi >= 8."""
    worst = 0.0
    for size, res in sorted(results.items()):
        print(f"\nL = {size}")
        print(f"{'T':>7} {'<E>/N':>16} {'exact':>9} {'dev/sigma':>10} {'L/xi':>7}")
        for t in res.temperatures:
            est = res.statistics(t).energy
            exact = onsager_energy(t)
            sigma = (est.value - exact) / est.error if est.error > 0 else math.inf
            ratio = size / correlation_length(t)
            large = ratio >= LARGE_LATTICE_RATIO
            flag = "" if large else "  (finite-size regime)"
            print(
                f"{t:7.4f} {str(est):>16} {exact:9.5f} {sigma:10.2f} {ratio:7.1f}{flag}"
            )
            if large:
                worst = max(worst, abs(sigma))
    return worst


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path.cwd(), help="output directory")
    parser.add_argument("--quick", action="store_true", help="tiny budget (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.quick:
        sizes, temps, n_sweeps, n_therm, budget = (
            QUICK_SIZES,
            QUICK_TEMPERATURES,
            2_000,
            200,
            "quick",
        )
    else:
        sizes, temps, n_sweeps, n_therm, budget = (
            FULL_SIZES,
            FULL_TEMPERATURES,
            20_000,
            2_000,
            f"{20_000 // 5} samples/T",
        )

    started = time.perf_counter()
    results = {
        size: run(
            size, temps, n_sweeps=n_sweeps, n_thermalization=n_therm, seed=args.seed + i
        )
        for i, size in enumerate(sizes)
    }
    elapsed = time.perf_counter() - started
    worst = report(results)
    figure = make_figure(results, args.out / "onsager_reproduction.png", budget=budget)
    print(
        f"\nworst deviation from Onsager at L/xi >= {LARGE_LATTICE_RATIO:.0f}: "
        f"{worst:.2f} sigma"
    )
    print(f"wrote {figure}  (simulation {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
