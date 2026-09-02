#!/usr/bin/env python3
"""Locate Tc of the square-lattice Ising model from Binder-cumulant crossings.

The Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2) of a finite periodic
L x L lattice is size-independent at the critical point to leading order
(Binder 1981): curves for different L cross at Tc. This script measures
U4(T) for three sizes with the Swendsen-Wang algorithm on a grid around the
exact Tc = 2 / ln(1 + sqrt 2) = 2.2692 (Onsager 1944), quotes jackknife
error bars from ``SimulationResults.statistics``, and reads the crossing
of each consecutive size pair off the difference D(T) = U4_L1(T) - U4_L2(T)
by linear interpolation between the two grid points that bracket its sign
change. The statistical error is the standard deviation of that root over
a parametric bootstrap (each D(T_i) redrawn from N(D_i, dD_i)).

The crossing of the largest pair typically lands within a few tenths of a
percent of the exact value; the remaining offset shrinks with L (see the
Tc campaign on the physics page for the full four-lattice study with a
quadratic fit and finite-size drift estimates: ``scripts/tc_campaign.py``).

Usage:
    python examples/tc_binder_crossing.py                 # full budget
    python examples/tc_binder_crossing.py --out figures/  # choose the directory
    python examples/tc_binder_crossing.py --quick         # seconds, smoke test

Runtime (Apple M4, release build): measured wall time is printed at the end.
"""

from __future__ import annotations

import argparse
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

FULL_SIZES: Final[tuple[int, ...]] = (16, 24, 32)
QUICK_SIZES: Final[tuple[int, ...]] = (8, 12)
HALF_WIDTH: Final = 0.03  # grid spans Tc * (1 +- HALF_WIDTH)
BOOTSTRAP_DRAWS: Final = 2_000

FloatArray = NDArray[np.float64]


def temperature_grid(n_points: int) -> tuple[float, ...]:
    """Evenly spaced temperatures within +-HALF_WIDTH of the exact Tc."""
    grid = TC_SQUARE_2D * np.linspace(1.0 - HALF_WIDTH, 1.0 + HALF_WIDTH, n_points)
    return tuple(float(t) for t in grid)


def run(
    size: int,
    temperatures: Sequence[float],
    *,
    n_sweeps: int,
    n_thermalization: int,
    seed: int,
) -> SimulationResults:
    """Swendsen-Wang in independent mode, one chain per temperature."""
    config = SimulationConfig(
        lattice=LatticeConfig(size=size, j1=1.0),
        algorithm=Algorithm.SWENDSEN_WANG,
        mode=ExecutionMode.INDEPENDENT,
        temperatures=tuple(temperatures),
        n_sweeps=n_sweeps,
        n_thermalization=n_thermalization,
        measurement_interval=2,
        store_configs=False,
        seed=seed,
    )
    return Simulation(config).run(show_progress=False)


def binder_curve(
    results: SimulationResults,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Temperatures, U4 values and their jackknife errors, in grid order."""
    temps = np.array(results.temperatures, dtype=np.float64)
    order = np.argsort(temps)
    estimates = [results.statistics(float(t)).binder_cumulant for t in temps[order]]
    values = np.array([e.value for e in estimates])
    errors = np.array([e.error for e in estimates])
    return temps[order], values, errors


def _interpolated_root(temps: FloatArray, diff: FloatArray) -> float:
    """Root of the piecewise-linear interpolant of diff(T); NaN if no sign change."""
    signs = np.sign(diff)
    change = np.nonzero(signs[:-1] * signs[1:] < 0)[0]
    if change.size == 0:
        return float("nan")
    i = int(change[np.argmin(np.abs(temps[change] - TC_SQUARE_2D))])
    t0, t1, d0, d1 = temps[i], temps[i + 1], diff[i], diff[i + 1]
    return float(t0 - d0 * (t1 - t0) / (d1 - d0))


def crossing(
    temps: FloatArray,
    u_small: FloatArray,
    e_small: FloatArray,
    u_large: FloatArray,
    e_large: FloatArray,
    *,
    seed: int,
) -> tuple[float, float]:
    """Crossing temperature of two U4 curves and its bootstrap error."""
    diff = u_small - u_large
    err = np.hypot(e_small, e_large)
    estimate = _interpolated_root(temps, diff)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            _interpolated_root(temps, rng.normal(diff, err))
            for _ in range(BOOTSTRAP_DRAWS)
        ]
    )
    draws = draws[np.isfinite(draws)]
    error = float(draws.std(ddof=1)) if draws.size > 1 else float("nan")
    return estimate, error


def make_figure(
    curves: dict[int, tuple[FloatArray, FloatArray, FloatArray]],
    crossings: dict[tuple[int, int], tuple[float, float]],
    output: Path,
    *,
    budget: str,
) -> Path:
    """U4(T) for every size, the exact Tc and the pairwise crossings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    markers = ("s", "o", "^", "D")
    for (size, (temps, values, errors)), marker in zip(sorted(curves.items()), markers):
        for ax in (ax_full, ax_zoom):
            ax.errorbar(
                temps,
                values,
                yerr=errors,
                fmt=f"{marker}-",
                ms=4,
                lw=1,
                capsize=2,
                label=f"$L={size}$",
            )
    for ax in (ax_full, ax_zoom):
        ax.axvline(TC_SQUARE_2D, color="grey", ls="--", lw=0.8, label="exact $T_c$")
        ax.set_xlabel("$T$  ($J = k_B = 1$)")
    ax_full.set_ylabel("Binder cumulant $U_4$")
    ax_full.set_title("Full grid", fontsize=10)
    ax_full.legend(frameon=False, fontsize=8)

    for k, ((small, large), (tc, err)) in enumerate(crossings.items()):
        if np.isfinite(tc):
            ax_zoom.axvline(tc, color="C3", lw=0.8, alpha=0.7)
            ax_zoom.axvspan(tc - err, tc + err, color="C3", alpha=0.10)
            ax_zoom.text(
                0.02,
                0.97 - 0.07 * k,
                f"$L={small},{large}$: $T_c = {tc:.4f}({round(err * 1e4):.0f})$",
                transform=ax_zoom.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="C3",
            )
    zoom_half = 0.012 * TC_SQUARE_2D
    ax_zoom.set_xlim(TC_SQUARE_2D - zoom_half, TC_SQUARE_2D + zoom_half)
    ax_zoom.set_ylim(0.58, 0.64)
    ax_zoom.set_title("Crossing region", fontsize=10)
    fig.suptitle(
        f"Binder-cumulant crossings, square lattice, Swendsen-Wang ({budget})",
        fontsize=10,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path.cwd(), help="output directory")
    parser.add_argument("--quick", action="store_true", help="tiny budget (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.quick:
        sizes, n_points, n_sweeps, n_therm, budget = QUICK_SIZES, 5, 2_000, 200, "quick"
    else:
        sizes, n_points, n_sweeps, n_therm, budget = (
            FULL_SIZES,
            13,
            20_000,
            2_000,
            "10 000 samples/T",
        )
    temps = temperature_grid(n_points)

    started = time.perf_counter()
    curves = {
        size: binder_curve(
            run(
                size,
                temps,
                n_sweeps=n_sweeps,
                n_thermalization=n_therm,
                seed=args.seed + 1000 * i,
            )
        )
        for i, size in enumerate(sizes)
    }
    elapsed = time.perf_counter() - started

    crossings: dict[tuple[int, int], tuple[float, float]] = {}
    ordered = sorted(curves)
    for small, large in zip(ordered[:-1], ordered[1:]):
        t, u_s, e_s = curves[small]
        _, u_l, e_l = curves[large]
        crossings[(small, large)] = crossing(t, u_s, e_s, u_l, e_l, seed=args.seed)

    print(f"exact Tc = {TC_SQUARE_2D:.5f}")
    for (small, large), (tc, err) in crossings.items():
        dev = (tc - TC_SQUARE_2D) / TC_SQUARE_2D * 100
        print(f"L = {small:2d}/{large:2d}: Tc = {tc:.4f} +- {err:.4f}  ({dev:+.2f} %)")
    figure = make_figure(
        curves, crossings, args.out / "tc_binder_crossing.png", budget=budget
    )
    print(f"wrote {figure}  (simulation {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
