#!/usr/bin/env python3
"""The first-order transition of the cubic J1-J2 Ising model by Wang-Landau sampling.

With a ferromagnetic nearest-neighbour coupling J1 = 1 and an
antiferromagnetic next-nearest-neighbour coupling J2 = -1/2 (twelve face
diagonals), the simple cubic lattice orders into ferromagnetic planes that
alternate along one axis — the layered phase, six ordered states (three
axes, two signs), e0 = -J1 + 2 J2 = -2 per site. The transition out of it
is first order: at the finite-size transition temperature the canonical
energy distribution has two peaks, one per phase, separated by a
free-energy barrier that grows with the interface area L^2 (Lee and
Kosterlitz, Phys. Rev. Lett. 65, 137 (1990)). Canonical samplers,
parallel tempering included, get trapped on one side of that barrier;
the flat-histogram walk of Wang and Landau (Phys. Rev. Lett. 86, 2050
(2001)) crosses it as often as it likes.

For each size the script estimates the density of states in the energy
window that holds both phases (eight replica-exchange windows in parallel
at the full budget), runs a multicanonical production stage,
reweights it to the equal-height temperature T_h(L) (the two peaks equally
high), and reports the equal-weight temperature T_w(L) with six ordered
states per disordered one (the finite-size estimate of T_c with the
smallest corrections, Borgs and Kotecky (1990)), the barrier
Delta F / T = ln(P_peak / P_bottom) and the interface tension
sigma = T Delta F/T / (2 L^2). The figure shows the reweighted energy
distributions at T_h(L) and the barrier against L^2. The barrier is only
resolved once the peaks separate (from L = 12 on); at the quick budget
(L = 4 and 6) the distributions are single-peaked and the script reports
the barrier as not available, which is the honest answer at that size.

Usage:
    python examples/cubic_first_order.py                 # full budget
    python examples/cubic_first_order.py --out figures/  # choose the directory
    python examples/cubic_first_order.py --quick         # seconds, smoke test

Runtime (Apple M4, release build): measured wall time is printed at the end.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from mcising import (
    LatticeConfig,
    LatticeType,
    WangLandauConfig,
    WangLandauResults,
    WangLandauSimulation,
)
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

J2: Final = -0.5
#: Per-site energies of the two phases at coexistence lie around -1.2 and
#: -1.0; the window holds both with a wide margin at every size below.
ENERGY_WINDOW: Final = (-1.7, -0.5)
#: The transition sits near T = 2.4; the pseudo-transition temperatures are
#: searched in this bracket.
BRACKET: Final = (2.2, 2.6)
FALLBACK_TEMPERATURE: Final = 2.4
#: Ordered states per disordered one for the equal-weight criterion.
Q_ORDERED: Final = 6.0
LAYERED_COLUMNS: Final = (1, 2, 4)


@dataclass(frozen=True)
class Budget:
    sizes: tuple[int, ...]
    log_f_final: float
    n_windows: int
    walkers_per_window: int
    n_walkers: int
    production_sweeps: int
    label: str


FULL_BUDGET: Final = Budget(
    sizes=(8, 12, 16),
    log_f_final=1e-5,
    n_windows=8,
    walkers_per_window=1,
    n_walkers=8,
    production_sweeps=60_000,
    label="8 windows, ln f -> 1e-5, 8 x 60 000 production sweeps",
)
QUICK_BUDGET: Final = Budget(
    sizes=(4, 6),
    log_f_final=1e-3,
    n_windows=1,
    walkers_per_window=1,
    n_walkers=2,
    production_sweeps=2_000,
    label="quick",
)


@dataclass(frozen=True)
class SizeResult:
    size: int
    results: WangLandauResults
    t_height: float
    t_height_error: float
    t_weight: float
    t_weight_error: float
    barrier: float
    barrier_error: float
    tension: float
    tension_error: float
    seconds: float

    @property
    def histogram_temperature(self) -> float:
        return self.t_height if math.isfinite(self.t_height) else FALLBACK_TEMPERATURE


def run_size(size: int, budget: Budget, *, seed: int) -> SizeResult:
    """Wang-Landau + production for one size, then the first-order analysis."""
    config = WangLandauConfig(
        lattice=LatticeConfig(lattice_type=LatticeType.CUBIC, size=size, j1=1.0, j2=J2),
        energy_window=ENERGY_WINDOW,
        log_f_final=budget.log_f_final,
        check_interval=200,
        exchange_interval=50,
        n_windows=budget.n_windows,
        walkers_per_window=budget.walkers_per_window,
        n_walkers=budget.n_walkers,
        production_sweeps=budget.production_sweeps,
        production_thermalization=budget.production_sweeps // 20,
        seed=seed,
    )
    started = time.perf_counter()
    results = WangLandauSimulation(config).run(show_progress=False)
    seconds = time.perf_counter() - started
    t_height = results.equal_height_temperature(BRACKET)
    t_weight = results.equal_weight_temperature(BRACKET, q=Q_ORDERED)
    if math.isfinite(t_height.value):
        barrier = results.free_energy_barrier(t_height.value)
        barrier_value, barrier_error = barrier.barrier.value, barrier.barrier.error
        tension, tension_error = (
            barrier.interface_tension.value,
            barrier.interface_tension.error,
        )
    else:
        barrier_value = barrier_error = tension = tension_error = math.nan
    return SizeResult(
        size=size,
        results=results,
        t_height=t_height.value,
        t_height_error=t_height.error,
        t_weight=t_weight.value,
        t_weight_error=t_weight.error,
        barrier=barrier_value,
        barrier_error=barrier_error,
        tension=tension,
        tension_error=tension_error,
        seconds=seconds,
    )


def _fmt(value: float, error: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    if not math.isfinite(error):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} +- {error:.{digits}f}"


def report(rows: Sequence[SizeResult]) -> None:
    for row in rows:
        wl, prod = row.results.wang_landau, row.results.production
        print(
            f"L = {row.size:2d}: WL {'converged' if wl.converged else 'NOT converged'} "
            f"in {wl.total_sweeps} sweeps ({wl.n_windows} window(s) x "
            f"{wl.walkers_per_window} walker(s)); production flatness "
            f"{prod.histogram_flatness:.2f}, round trips {list(prod.round_trips)}; "
            f"{row.seconds:.1f} s"
        )
        estimate = row.results.reweight(
            row.histogram_temperature, order_parameter=LAYERED_COLUMNS
        )
        print(
            f"        T_h = {_fmt(row.t_height, row.t_height_error)}   "
            f"T_w(q=6) = {_fmt(row.t_weight, row.t_weight_error)}   "
            f"Delta F/T = {_fmt(row.barrier, row.barrier_error, 3)}   "
            f"sigma = {_fmt(row.tension, row.tension_error, 5)}   "
            f"psi(T) = {estimate.order_parameter}   "
            f"n_eff = {estimate.effective_samples:.0f}"
        )


def make_figure(rows: Sequence[SizeResult], output: Path, *, budget: str) -> Path:
    """P_T(E) at the equal-height temperature per size, and the barrier vs L^2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_hist, ax_barrier) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for row in rows:
        energies, probability = row.results.canonical_energy_histogram(
            row.histogram_temperature
        )
        # The coexistence region: bins holding at least a thousandth of the
        # peak, normalised to the peak so different sizes share one axis.
        keep = probability > 1e-3 * probability.max()
        label = f"$L={row.size}$, $T={row.histogram_temperature:.3f}$"
        ax_hist.plot(
            energies[keep], probability[keep] / probability.max(), lw=1.2, label=label
        )
    ax_hist.set_xlabel("$E/N$  ($J_1 = 1$, $J_2 = -1/2$)")
    ax_hist.set_ylabel("$P_T(E) / P_\\mathrm{max}$")
    ax_hist.set_ylim(0.0, 1.08)
    ax_hist.set_title("Canonical energy distribution at $T_h(L)$", fontsize=10)
    ax_hist.legend(frameon=False, fontsize=8, loc="lower center")

    sizes = np.array([row.size for row in rows], dtype=float)
    barriers = np.array([row.barrier for row in rows])
    errors = np.array([row.barrier_error for row in rows])
    finite = np.isfinite(barriers)
    if finite.any():
        ax_barrier.errorbar(
            sizes[finite] ** 2,
            barriers[finite],
            yerr=np.where(np.isfinite(errors[finite]), errors[finite], 0.0),
            fmt="o",
            capsize=3,
            color="C3",
        )
        for size, barrier in zip(sizes[finite], barriers[finite]):
            ax_barrier.annotate(
                f"$L={size:.0f}$",
                (size * size, barrier),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
    else:
        ax_barrier.text(
            0.5,
            0.5,
            "no double peak at these sizes",
            ha="center",
            va="center",
            transform=ax_barrier.transAxes,
            fontsize=9,
        )
    ax_barrier.set_xlabel("$L^2$ (interface area)")
    ax_barrier.set_ylabel(r"$\Delta F / T = \ln(P_\mathrm{peak} / P_\mathrm{bottom})$")
    ax_barrier.set_title("Free-energy barrier", fontsize=10)
    fig.suptitle(
        f"Cubic $J_1$-$J_2$ Ising model, layered transition ({budget})", fontsize=10
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
    budget = QUICK_BUDGET if args.quick else FULL_BUDGET

    started = time.perf_counter()
    rows = [
        run_size(size, budget, seed=args.seed + 100 * index)
        for index, size in enumerate(budget.sizes)
    ]
    elapsed = time.perf_counter() - started
    report(rows)
    figure = make_figure(rows, args.out / "cubic_first_order.png", budget=budget.label)
    print(f"wrote {figure}  (simulation {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
