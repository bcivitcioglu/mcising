#!/usr/bin/env python3
"""Tc campaign: measure the critical temperature on every 2D/3D lattice.

For each lattice the Binder cumulant ``U4 = 1 - <m^4>/(3 <m^2>^2)`` and the
specific heat are measured on a grid of temperatures around the literature
Tc for several linear sizes L. The headline estimate is the crossing of the
U4(T) curves of the two largest sizes (Binder 1981); the specific-heat peak
temperature at each L is reported alongside as a secondary, finite-size
shifted estimator.

Sampling: Swendsen-Wang in independent mode (every temperature is its own
chain from a random start, all temperatures in parallel). SW is used
throughout, not Wolff: off-critical a single Wolff cluster flips only a few
dozen spins and at a fixed small cluster interval the series stays so
correlated that the quoted error is not honest (see the P14 Onsager tests).

Uncertainties: U4 and Cv carry delete-one-block jackknife errors from
``mcising.statistics``. The crossing is the root of a weighted quadratic
fitted to ``D(T) = U4_{L1}(T) - U4_{L2}(T)`` over the grid points within
CROSSING_HALF_SPAN of the literature Tc (a straight line is measurably
mis-specified: the scaling variable reaches |x| ~ 1 across the window for
the largest L, and the resulting root bias is several statistical errors).
The statistical error is the standard deviation of the root over a
parametric bootstrap (each D(T_i) redrawn from N(D_i, dD_i)). The Cv peak
is the vertex of a weighted parabola over the points within PEAK_HALF_SPAN
of the measured maximum, its error half the 16-84 percentile interval of
the same bootstrap (a ratio estimator, so a percentile interval rather
than a standard deviation). The finite-size drift between the last two
crossings, ``|Tc(L3,L4) - Tc(L2,L3)|``, is quoted as a separate
systematic. Every fit reports chi^2/dof: ~1 means the jackknife errors
are honest and the local polynomial is adequate.

Usage:
    uv run python scripts/tc_campaign.py               # full budget (~5 min)
    uv run python scripts/tc_campaign.py --quick       # the slow test's budget
    uv run python scripts/tc_campaign.py --write-docs  # also refresh the docs table
    uv run python scripts/tc_campaign.py --from-json scripts/tc_campaign_results.json \\
        --write-docs                                   # re-render without running

The committed ``scripts/tc_campaign_results.json`` is what
``tests/test_tc_campaign.py`` checks against ``mcising.constants``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import mcising
import numpy as np
from mcising._provenance import git_commit
from mcising.config import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.constants import (
    TC_CUBIC_3D,
    TC_HONEYCOMB_2D,
    TC_SQUARE_2D,
    TC_TRIANGULAR_2D,
)
from mcising.simulation import Simulation
from mcising.statistics import binder_cumulant
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = REPO_ROOT / "scripts" / "tc_campaign_results.json"
DOCS_PAGE: Final = REPO_ROOT / "docs" / "advanced" / "physics.md"
DOCS_BEGIN: Final = "<!-- tc-campaign:begin -->"
DOCS_END: Final = "<!-- tc-campaign:end -->"
SCHEMA_VERSION: Final = 1

#: Temperature grid, as fractions of the literature Tc: a coarse grid over
#: +-GRID_HALF_WIDTH at GRID_STEP (brackets every specific-heat peak used
#: here — the cubic L=8 peak sits ~3% below Tc, the 2D L=16 peaks ~2%
#: above) plus a fine grid over +-FINE_HALF_WIDTH at FINE_STEP where the
#: Binder curves cross. 27 temperatures in all.
GRID_HALF_WIDTH: Final = 0.05
GRID_STEP: Final = 0.005
FINE_HALF_WIDTH: Final = 0.015
FINE_STEP: Final = 0.0025
#: Points within this fraction of Tc_lit enter the quadratic fit of D(T)
#: (9 points at the fine spacing).
CROSSING_HALF_SPAN: Final = 0.010
#: Points within this fraction of the measured Cv maximum enter its
#: parabola fit.
PEAK_HALF_SPAN: Final = 0.015
BOOTSTRAP_DRAWS: Final = 2000
BOOTSTRAP_SEED: Final = 20260901
#: Seed spacing between (lattice, size) runs. Independent mode derives the
#: per-temperature stream from ``seed + temperature_index`` (index < 27), so
#: a stride of 1000 keeps every chain in the campaign on a distinct stream.
SEED_STRIDE: Final = 1000
#: Stationarity canary: |U4(first half) - U4(second half)| beyond this many
#: full-series jackknife errors is flagged. Each half carries ~sqrt(2) x the
#: full error and the difference ~2x, so 8 full errors = 4 sigma.
STATIONARITY_LIMIT: Final = 8.0
#: Fraction of crossing bootstrap draws allowed to have no root at all;
#: above this the standard deviation of the rest would be censored.
MAX_FAILED_DRAW_FRACTION: Final = 0.01
#: Fraction of peak bootstrap draws allowed to diverge (no maximum). The
#: 16-84 percentile interval absorbs divergent replicas in its tails, so
#: it stays finite and honest up to this fraction on either side.
MAX_DIVERGENT_PEAK_FRACTION: Final = 0.16
MEASUREMENT_INTERVAL: Final = 2


class CampaignError(RuntimeError):
    """The data cannot support the estimate (grid, statistics, or fit)."""


@dataclass(frozen=True)
class LatticeSpec:
    """One lattice of the campaign: sizes and the literature Tc."""

    name: str
    lattice_type: LatticeType
    sizes: tuple[int, ...]
    tc_exact: float
    tc_source: str


LATTICES: Final[tuple[LatticeSpec, ...]] = (
    LatticeSpec(
        "square",
        LatticeType.SQUARE,
        (16, 24, 32, 48, 64),
        TC_SQUARE_2D,
        "Onsager (1944), exact: 2/ln(1+sqrt(2))",
    ),
    LatticeSpec(
        "triangular",
        LatticeType.TRIANGULAR,
        (16, 24, 32, 48, 64),
        TC_TRIANGULAR_2D,
        "Houtappel (1950), exact: 4/ln(3)",
    ),
    LatticeSpec(
        "honeycomb",
        LatticeType.HONEYCOMB,
        (16, 24, 32, 48),
        TC_HONEYCOMB_2D,
        "Houtappel (1950), exact: 2/ln(2+sqrt(3))",
    ),
    LatticeSpec(
        "cubic",
        LatticeType.CUBIC,
        (8, 12, 16, 24),
        TC_CUBIC_3D,
        "Ferrenberg, Xu & Landau (2018), MC: 1/0.221654626(5)",
    ),
)


@dataclass(frozen=True)
class Budget:
    """Sweep schedule per (lattice, size) run; every temperature gets it."""

    n_thermalization: int
    n_sweeps: int
    measurement_interval: int
    quick: bool


FULL_BUDGET: Final = Budget(2000, 40_000, MEASUREMENT_INTERVAL, quick=False)
QUICK_BUDGET: Final = Budget(1000, 8_000, MEASUREMENT_INTERVAL, quick=True)


@dataclass(frozen=True)
class Crossing:
    """Binder crossing of one size pair; ``tc`` is None when the data do not
    support it (``reason`` says why — only allowed for diagnostic pairs)."""

    sizes: tuple[int, int]
    tc: float | None
    error: float | None
    chi2_dof: float | None
    window: tuple[float, float] | None
    n_points: int
    n_failed_draws: int
    reason: str | None


@dataclass(frozen=True)
class Peak:
    """Specific-heat peak temperature at one size; ``tc`` is None when the
    grid does not bracket the peak (``reason`` says why)."""

    size: int
    tc: float | None
    error: float | None
    chi2_dof: float | None
    window: tuple[float, float] | None
    n_points: int
    n_failed_draws: int
    reason: str | None


# --- analysis -------------------------------------------------------------


def _as_array(name: str, values: Sequence[float] | FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        msg = f"{name}: empty or non-finite values"
        raise CampaignError(msg)
    return arr


def _check_errors(name: str, err: FloatArray) -> None:
    if not np.all(err > 0.0):
        msg = f"{name}: every error must be finite and > 0, got {err}"
        raise CampaignError(msg)


def _span_window(temps: FloatArray, centre: float, half_span: float) -> FloatArray:
    """Boolean mask of the points within ``half_span`` (relative) of ``centre``."""
    return np.asarray(np.abs(temps / centre - 1.0) <= half_span + 1e-9)


def _quadratic_roots(coef: FloatArray) -> FloatArray:
    """Real roots of ``a u^2 + b u + c`` per column of a (3, K) array.

    Returns shape (2, K) with ``nan`` where a root does not exist; a
    numerically vanishing quadratic term falls back to the single linear
    root.
    """
    a, b, c = coef
    roots = np.full((2, a.size), np.nan)
    linear = np.abs(a) < 1e-12 * np.maximum(1.0, np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        roots[0, linear] = -c[linear] / b[linear]
        disc = b * b - 4.0 * a * c
        quad = ~linear & (disc >= 0.0)
        sq = np.sqrt(np.where(quad, disc, 0.0))
        roots[0, quad] = (-b[quad] - sq[quad]) / (2.0 * a[quad])
        roots[1, quad] = (-b[quad] + sq[quad]) / (2.0 * a[quad])
    return roots


def _nearest_root(roots: FloatArray, target: float) -> FloatArray:
    """Per column, the real root closest to ``target`` (``nan`` if none)."""
    dist = np.where(np.isfinite(roots), np.abs(roots - target), np.inf)
    pick = np.where(dist[0] <= dist[1], roots[0], roots[1])
    any_root = np.isfinite(dist).any(axis=0)
    return np.asarray(np.where(any_root, pick, np.nan), dtype=np.float64)


def binder_crossing(
    temps: Sequence[float] | FloatArray,
    u_small: Sequence[float] | FloatArray,
    err_small: Sequence[float] | FloatArray,
    u_large: Sequence[float] | FloatArray,
    err_large: Sequence[float] | FloatArray,
    *,
    tc_ref: float,
    sizes: tuple[int, int] = (0, 0),
    half_span: float = CROSSING_HALF_SPAN,
    draws: int = BOOTSTRAP_DRAWS,
    rng: np.random.Generator | None = None,
) -> Crossing:
    """Crossing temperature of two Binder-cumulant curves.

    ``D(T) = U4_small(T) - U4_large(T)`` is negative below Tc (the larger
    lattice is closer to 2/3) and positive above (its cumulant decays
    faster), so its root is the crossing. A weighted quadratic in
    ``u = 100 (T / tc_ref - 1)`` is fitted to the grid points within
    ``half_span`` of ``tc_ref`` and the unique real root inside that window
    is the estimate; no unique root means the grid does not bracket the
    crossing and a :class:`CampaignError` is raised rather than an
    extrapolation quoted. The error is the standard deviation of the root
    over ``draws`` parametric-bootstrap replicas (D_i ~ N(D_i, dD_i), which
    is exactly what redrawing the two independent U4 values gives).

    ``tc_ref`` only centres the data window; it does not enter the estimate.
    """
    t = _as_array("temps", temps)
    us, es = _as_array("u_small", u_small), _as_array("err_small", err_small)
    ul, el = _as_array("u_large", u_large), _as_array("err_large", err_large)
    if not (t.size == us.size == es.size == ul.size == el.size):
        msg = "temps and both cumulant series must have equal length"
        raise CampaignError(msg)
    mask = _span_window(t, tc_ref, half_span)
    n_points = int(mask.sum())
    label = f"Binder crossing L={sizes[0]}/{sizes[1]}"
    if n_points < 5:
        msg = f"{label}: only {n_points} grid points within {half_span:.3%} of Tc"
        raise CampaignError(msg)
    u = 100.0 * (t[mask] / tc_ref - 1.0)
    d = us[mask] - ul[mask]
    err = np.hypot(es[mask], el[mask])
    _check_errors(f"{label} errors", err)
    weights = 1.0 / err
    lo, hi = float(u[0]), float(u[-1])

    def fit_roots(values: FloatArray) -> FloatArray:
        coef = np.polyfit(u, values.T, 2, w=weights)
        return _quadratic_roots(np.asarray(coef, dtype=np.float64))

    coef = np.polyfit(u, d, 2, w=weights)
    chi2 = float(np.sum(((d - np.polyval(coef, u)) / err) ** 2))
    chi2_dof = chi2 / (n_points - 3)
    window = (float(t[mask][0]), float(t[mask][-1]))
    central = fit_roots(d[None, :])[:, 0]
    inside = central[(central >= lo) & (central <= hi)]
    if inside.size != 1:
        msg = (
            f"{label}: {inside.size} roots of the quadratic fit inside "
            f"[{window[0]:.5f}, {window[1]:.5f}], need exactly one (D={d}, dD={err})"
        )
        raise CampaignError(msg)
    u_root = float(inside[0])
    tc = tc_ref * (1.0 + u_root / 100.0)

    # Replicas keep the real root nearest the central estimate wherever it
    # falls: confining them to the window would censor the spread and
    # understate the error. A replica fails only when its parabola has no
    # real root at all.
    gen = np.random.default_rng(BOOTSTRAP_SEED) if rng is None else rng
    draws_d = gen.normal(d, err, size=(draws, d.size))
    replicas = _nearest_root(fit_roots(draws_d), u_root)
    finite = np.isfinite(replicas)
    n_failed = int(draws - finite.sum())
    if n_failed > MAX_FAILED_DRAW_FRACTION * draws:
        msg = (
            f"{label}: {n_failed}/{draws} bootstrap draws have no crossing at "
            "all — increase the statistics"
        )
        raise CampaignError(msg)
    error = tc_ref * float(np.std(replicas[finite], ddof=1)) / 100.0
    return Crossing(sizes, tc, error, chi2_dof, window, n_points, n_failed, None)


def cv_peak(
    temps: Sequence[float] | FloatArray,
    cv: Sequence[float] | FloatArray,
    err: Sequence[float] | FloatArray,
    *,
    size: int = 0,
    half_span: float = PEAK_HALF_SPAN,
    draws: int = BOOTSTRAP_DRAWS,
    rng: np.random.Generator | None = None,
) -> Peak:
    """Temperature of the specific-heat maximum from a local parabola.

    A weighted quadratic in ``u = 100 (T / T_max - 1)`` is fitted to the
    grid points within ``half_span`` of the largest measured Cv; the vertex
    is the estimate. The error is half the 16-84 percentile interval of the
    vertex over the parametric bootstrap. When the maximum sits at the
    edge of the grid, or the vertex falls outside the fitted window, no
    estimate is quoted (``tc=None`` with ``reason``) — the Cv column is
    secondary and must never be extrapolated.
    """
    t = _as_array("temps", temps)
    c, e = _as_array("cv", cv), _as_array("err", err)
    if not (t.size == c.size == e.size):
        msg = "temps, cv and err must have equal length"
        raise CampaignError(msg)
    _check_errors("cv errors", e)
    i_max = int(np.argmax(c))
    t_max = float(t[i_max])
    mask = _span_window(t, t_max, half_span)
    n_below = int(mask[:i_max].sum())
    n_above = int(mask[i_max + 1 :].sum())
    n_points = n_below + n_above + 1
    if n_below < 2 or n_above < 2:
        reason = (
            f"Cv maximum at T={t_max:.5f} has only {n_below} grid points below "
            f"and {n_above} above within {half_span:.1%}; the grid does not "
            "bracket the peak"
        )
        return Peak(size, None, None, None, None, n_points, 0, reason)
    u = 100.0 * (t[mask] / t_max - 1.0)
    y, ey = c[mask], e[mask]
    w = 1.0 / ey
    lo, hi = float(u[0]), float(u[-1])

    def vertices(values: FloatArray) -> FloatArray:
        coef = np.polyfit(u, values.T, 2, w=w)
        c2, c1, _ = np.asarray(coef, dtype=np.float64)
        out = np.full(c2.shape, np.nan)
        ok = c2 < 0.0
        out[ok] = -c1[ok] / (2.0 * c2[ok])
        # A parabola without a maximum (c2 >= 0) peaks beyond the window on
        # the side it rises towards: +-inf. The percentile interval places
        # such replicas in its tails instead of dropping them, so it stays
        # honest as long as fewer than 16% diverge on either side.
        out[~ok] = np.where(c1[~ok] > 0.0, np.inf, -np.inf)
        return np.asarray(out, dtype=np.float64)

    u_vertex = float(vertices(y[None, :])[0])
    coef = np.polyfit(u, y, 2, w=w)
    chi2 = float(np.sum(((y - np.polyval(coef, u)) / ey) ** 2))
    chi2_dof = chi2 / (n_points - 3)
    window = (float(t[mask][0]), float(t[mask][-1]))
    # The central vertex must lie inside the fitted window (no
    # extrapolation); replicas are unconstrained so the spread is honest.
    if not math.isfinite(u_vertex) or not (lo <= u_vertex <= hi):
        reason = (
            f"parabola through the {n_points} points around T={t_max:.5f} has "
            f"no maximum inside [{window[0]:.5f}, {window[1]:.5f}]"
        )
        return Peak(size, None, None, chi2_dof, window, n_points, 0, reason)
    tc = t_max * (1.0 + u_vertex / 100.0)

    gen = np.random.default_rng(BOOTSTRAP_SEED) if rng is None else rng
    replicas = vertices(gen.normal(y, ey, size=(draws, y.size)))
    n_failed = int(np.sum(~np.isfinite(replicas)))
    p16, p84 = np.percentile(replicas, [16.0, 84.0])
    if n_failed > MAX_DIVERGENT_PEAK_FRACTION * draws or not (
        math.isfinite(p16) and math.isfinite(p84)
    ):
        reason = (
            f"{n_failed}/{draws} bootstrap draws have no maximum (curvature unresolved)"
        )
        return Peak(size, None, None, chi2_dof, window, n_points, n_failed, reason)
    error = t_max * float(p84 - p16) / 200.0
    return Peak(size, tc, error, chi2_dof, window, n_points, n_failed, None)


# --- campaign --------------------------------------------------------------


def temperature_grid(tc_exact: float) -> tuple[float, ...]:
    """The campaign grid around a literature Tc (see the GRID_* constants)."""
    coarse = np.arange(-GRID_HALF_WIDTH, GRID_HALF_WIDTH + GRID_STEP / 2, GRID_STEP)
    fine = np.arange(-FINE_HALF_WIDTH, FINE_HALF_WIDTH + FINE_STEP / 2, FINE_STEP)
    factors = np.unique(np.round(np.concatenate([coarse, fine]), 6))
    return tuple(float(x) for x in np.round(tc_exact * (1.0 + factors), 10))


def _seed_block(spec: LatticeSpec) -> int:
    """Index of the lattice's first (lattice, size) run in the global seed
    sequence — fixed by LATTICES so a partial run reproduces the streams
    of the full campaign."""
    offset = 0
    for other in LATTICES:
        if other.name == spec.name:
            return offset
        offset += len(other.sizes)
    return offset  # a spec outside LATTICES gets the next free block


def run_size(
    spec: LatticeSpec,
    size: int,
    temps: tuple[float, ...],
    budget: Budget,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run one (lattice, size) over the grid and tabulate U4 and Cv."""
    config = SimulationConfig(
        lattice=LatticeConfig(lattice_type=spec.lattice_type, size=size),
        algorithm=Algorithm.SWENDSEN_WANG,
        mode=ExecutionMode.INDEPENDENT,
        temperatures=temps,
        n_sweeps=budget.n_sweeps,
        n_thermalization=budget.n_thermalization,
        measurement_interval=budget.measurement_interval,
        store_configs=False,
        seed=seed,
    )
    start = time.perf_counter()
    results = Simulation(config).run(show_progress=False)
    elapsed = time.perf_counter() - start

    u4: list[float] = []
    u4_err: list[float] = []
    cv: list[float] = []
    cv_err: list[float] = []
    tau: list[float] = []
    flags = 0
    n_samples = 0
    for t in temps:
        stats = results.statistics(t)
        u4.append(stats.binder_cumulant.value)
        u4_err.append(stats.binder_cumulant.error)
        cv.append(stats.specific_heat.value)
        cv_err.append(stats.specific_heat.error)
        tau.append(stats.tau_int)
        n_samples = stats.n_samples
        m = results.magnetization[t]
        half = m.size // 2
        gap = abs(binder_cumulant(m[:half]) - binder_cumulant(m[half:]))
        if gap > STATIONARITY_LIMIT * stats.binder_cumulant.error:
            flags += 1
    return {
        "size": size,
        "num_sites": config.lattice.num_sites,
        "seed": seed,
        "n_samples": n_samples,
        "elapsed_seconds": elapsed,
        "binder": {"value": u4, "error": u4_err},
        "specific_heat": {"value": cv, "error": cv_err},
        "tau_int_energy": tau,
        "stationarity_flags": flags,
    }


def run_lattice(
    spec: LatticeSpec,
    budget: Budget,
    *,
    base_seed: int,
    draws: int = BOOTSTRAP_DRAWS,
    rng: np.random.Generator | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every size of one lattice and extract the Tc estimates."""
    temps = temperature_grid(spec.tc_exact)
    block = _seed_block(spec)
    per_size: list[dict[str, Any]] = []
    for j, size in enumerate(spec.sizes):
        seed = base_seed + SEED_STRIDE * (block + j)
        row = run_size(spec, size, temps, budget, seed=seed)
        per_size.append(row)
        log(
            f"  {spec.name:10s} L={size:3d} N={row['num_sites']:6d} "
            f"{row['elapsed_seconds']:6.1f}s  {row['n_samples']} samples/T  "
            f"tau_E<={max(row['tau_int_energy']):.2f}  "
            f"stationarity flags {row['stationarity_flags']}"
        )
    return analyse_lattice(spec, temps, per_size, draws=draws, rng=rng, log=log)


def analyse_lattice(
    spec: LatticeSpec,
    temps: Sequence[float],
    per_size: Sequence[dict[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    rng: np.random.Generator | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Crossings, peaks and the headline estimate from the per-size tables.

    Pure post-processing of the Monte Carlo tables (U4 and Cv with their
    errors per size and temperature), so :func:`reanalyse` can rebuild
    every estimate of a stored document without re-simulating.
    """
    crossings: list[Crossing] = []
    pairs = list(zip(per_size, per_size[1:]))
    for k, (a, b) in enumerate(pairs):
        sizes = (int(a["size"]), int(b["size"]))
        try:
            crossings.append(
                binder_crossing(
                    temps,
                    a["binder"]["value"],
                    a["binder"]["error"],
                    b["binder"]["value"],
                    b["binder"]["error"],
                    tc_ref=spec.tc_exact,
                    sizes=sizes,
                    draws=draws,
                    rng=rng,
                )
            )
        except CampaignError as exc:
            if k == len(pairs) - 1:
                raise  # the headline pair must resolve
            # A diagnostic pair may legitimately cross outside the window
            # (small L); record why instead of aborting the campaign.
            crossings.append(Crossing(sizes, None, None, None, None, 0, 0, str(exc)))
    peaks = [
        cv_peak(
            temps,
            row["specific_heat"]["value"],
            row["specific_heat"]["error"],
            size=int(row["size"]),
            draws=draws,
            rng=rng,
        )
        for row in per_size
    ]
    headline = crossings[-1]
    assert headline.tc is not None and headline.error is not None
    previous = [c.tc for c in crossings[:-1] if c.tc is not None]
    error_sys = abs(headline.tc - previous[-1]) if previous else math.nan
    deviation = 100.0 * (headline.tc - spec.tc_exact) / spec.tc_exact
    log(
        f"  {spec.name:10s} Tc = {headline.tc:.5f} ± {headline.error:.5f} (stat) "
        f"± {error_sys:.5f} (drift)  vs {spec.tc_exact:.5f}  ({deviation:+.3f}%)  "
        f"chi2/dof {headline.chi2_dof:.2f}"
    )
    return {
        "lattice_type": spec.lattice_type.value,
        "tc_exact": spec.tc_exact,
        "tc_source": spec.tc_source,
        "sizes": list(spec.sizes),
        "temperatures": [float(x) for x in temps],
        "runs": list(per_size),
        "crossings": [asdict(c) for c in crossings],
        "cv_peaks": [asdict(p) for p in peaks],
        "tc_measured": {
            "method": "binder_crossing",
            "sizes": list(headline.sizes),
            "value": headline.tc,
            "error_stat": headline.error,
            "error_sys": error_sys,
            "chi2_dof": headline.chi2_dof,
        },
        "deviation_percent": deviation,
        "stationarity_flags": sum(int(row["stationarity_flags"]) for row in per_size),
    }


def reanalyse(
    document: dict[str, Any],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Recompute every estimate of a results document from its stored tables.

    The Monte Carlo tables are kept verbatim together with the run
    provenance; crossings, peaks and the headline values are rebuilt with
    the current estimator, reference constants and bootstrap settings, so
    an estimator refinement never requires re-simulating.
    ``reanalysed_utc`` records the recomputation.
    """
    by_name = {spec.name: spec for spec in LATTICES}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for name, lat in list(document["lattices"].items()):
        spec = by_name.get(name)
        if spec is None:
            msg = f"unknown lattice {name!r} in the results document"
            raise CampaignError(msg)
        document["lattices"][name] = analyse_lattice(
            spec, lat["temperatures"], lat["runs"], draws=draws, rng=rng, log=log
        )
    document["budget"].update(
        {
            "crossing_half_span": CROSSING_HALF_SPAN,
            "peak_half_span": PEAK_HALF_SPAN,
            "bootstrap_draws": draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
    )
    document["reanalysed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    )
    return document


def run_campaign(
    specs: Sequence[LatticeSpec] = LATTICES,
    budget: Budget = FULL_BUDGET,
    *,
    base_seed: int = 42,
    draws: int = BOOTSTRAP_DRAWS,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the campaign and return the JSON-serialisable results document."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    start = time.perf_counter()
    lattices: dict[str, Any] = {}
    for spec in specs:
        grid = temperature_grid(spec.tc_exact)
        log(
            f"{spec.name}: {len(grid)} temperatures in [{grid[0]:.4f}, "
            f"{grid[-1]:.4f}], sizes {list(spec.sizes)}"
        )
        lattices[spec.name] = run_lattice(
            spec, budget, base_seed=base_seed, draws=draws, rng=rng, log=log
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "mcising_version": mcising.__version__,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "elapsed_seconds": time.perf_counter() - start,
        "budget": {
            **asdict(budget),
            "algorithm": Algorithm.SWENDSEN_WANG.value,
            "mode": ExecutionMode.INDEPENDENT.value,
            "base_seed": base_seed,
            "seed_stride": SEED_STRIDE,
            "grid_half_width": GRID_HALF_WIDTH,
            "grid_step": GRID_STEP,
            "fine_half_width": FINE_HALF_WIDTH,
            "fine_step": FINE_STEP,
            "crossing_half_span": CROSSING_HALF_SPAN,
            "peak_half_span": PEAK_HALF_SPAN,
            "bootstrap_draws": draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "lattices": lattices,
    }


# --- rendering ---------------------------------------------------------------


def _fmt_pm(value: float | None, error: float | None) -> str:
    if value is None or error is None:
        return "n/a"
    return f"{value:.4f} ± {error:.4f}"


def render_markdown_table(document: dict[str, Any]) -> str:
    """Render the results document as the docs' Markdown block.

    Deterministic formatting: this one function feeds both
    ``docs/advanced/physics.md`` and the docs-consistency test.
    """
    lines = [
        "| Lattice | L | Tc, Binder crossing (± stat ± drift) | "
        "Cv peak at largest L | Reference Tc | Δ (%) |",
        "|---|---|---|---|---|---|",
    ]
    for name, lat in document["lattices"].items():
        tc = lat["tc_measured"]
        sizes = ", ".join(str(s) for s in lat["sizes"])
        drift = tc["error_sys"]
        drift_txt = (
            f"{drift:.4f}"
            if isinstance(drift, float) and math.isfinite(drift)
            else "n/a"
        )
        peak = lat["cv_peaks"][-1]
        lines.append(
            f"| {name} | {sizes} "
            f"| {tc['value']:.4f} ± {tc['error_stat']:.4f} ± {drift_txt} "
            f"(L={tc['sizes'][0]}, {tc['sizes'][1]}) "
            f"| {_fmt_pm(peak['tc'], peak['error'])} (L={peak['size']}) "
            f"| {lat['tc_exact']:.5f} | {lat['deviation_percent']:+.2f} |"
        )
    lines += [
        "",
        "| Lattice | L pair | Crossing Tc (± stat) | χ²/dof |",
        "|---|---|---|---|",
    ]
    for name, lat in document["lattices"].items():
        for c in lat["crossings"]:
            if c["tc"] is None:
                cell, chi = "n/a (not bracketed)", "—"
            else:
                cell = f"{c['tc']:.4f} ± {c['error']:.4f}"
                chi = f"{c['chi2_dof']:.2f}"
            lines.append(
                f"| {name} | {c['sizes'][0]}, {c['sizes'][1]} | {cell} | {chi} |"
            )
    lines += [
        "",
        "| Lattice | L | Cv peak T (± stat) | χ²/dof |",
        "|---|---|---|---|",
    ]
    for name, lat in document["lattices"].items():
        for p in lat["cv_peaks"]:
            if p["tc"] is None:
                cell, chi = "n/a (not bracketed)", "—"
            else:
                cell = f"{p['tc']:.4f} ± {p['error']:.4f}"
                chi = f"{p['chi2_dof']:.2f}"
            lines.append(f"| {name} | {p['size']} | {cell} | {chi} |")
    budget = document["budget"]
    commit = document.get("git_commit")
    lines += [
        "",
        f"Swendsen–Wang, independent mode; {budget['n_thermalization']} "
        f"thermalization + {budget['n_sweeps']} measurement sweeps per "
        f"temperature, sampled every {budget['measurement_interval']} sweeps; "
        f"grid Tc·[1 ± {budget['grid_half_width']}] at {budget['grid_step']} "
        f"plus Tc·[1 ± {budget['fine_half_width']}] at {budget['fine_step']}; "
        f"crossing fit within ±{budget['crossing_half_span']}, peak fit within "
        f"±{budget['peak_half_span']} of the maximum; "
        f"mcising {document['mcising_version']}"
        + (f" ({commit[:7]})" if isinstance(commit, str) and commit else "")
        + f", generated {document['generated_utc']}"
        + (
            f", estimates recomputed {document['reanalysed_utc']}"
            if document.get("reanalysed_utc")
            else ""
        )
        + ".",
    ]
    return "\n".join(lines)


def write_docs_block(path: Path, block: str) -> None:
    """Replace the marker-delimited block in the docs page."""
    text = path.read_text(encoding="utf-8")
    if text.count(DOCS_BEGIN) != 1 or text.count(DOCS_END) != 1:
        msg = f"{path} must contain exactly one {DOCS_BEGIN} / {DOCS_END} pair"
        raise CampaignError(msg)
    head, rest = text.split(DOCS_BEGIN, 1)
    _, tail = rest.split(DOCS_END, 1)
    path.write_text(f"{head}{DOCS_BEGIN}\n{block}\n{DOCS_END}{tail}", encoding="utf-8")


# --- CLI --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced sweep budget (the slow test's setting)",
    )
    parser.add_argument(
        "--lattices",
        default=",".join(spec.name for spec in LATTICES),
        help="comma-separated subset of " + ", ".join(spec.name for spec in LATTICES),
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed (default 42)")
    parser.add_argument(
        "--draws", type=int, default=BOOTSTRAP_DRAWS, help="bootstrap draws"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="results JSON path"
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="skip the simulations and render an existing results JSON",
    )
    parser.add_argument(
        "--reanalyse",
        action="store_true",
        help="with --from-json: recompute crossings, peaks and headline values "
        "from the stored tables and write the result to --output",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="rewrite the table between the markers in "
        + str(DOCS_PAGE.relative_to(REPO_ROOT)),
    )
    args = parser.parse_args(argv)

    if args.from_json is not None:
        document = json.loads(args.from_json.read_text(encoding="utf-8"))
        if args.reanalyse:
            document = reanalyse(document, draws=args.draws)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(document, indent=1) + "\n", encoding="utf-8"
            )
            print(f"\nreanalysed {args.from_json} -> {args.output}")
    else:
        wanted = [name.strip() for name in args.lattices.split(",") if name.strip()]
        by_name = {spec.name: spec for spec in LATTICES}
        unknown = [name for name in wanted if name not in by_name]
        if unknown:
            parser.error(f"unknown lattice(s): {', '.join(unknown)}")
        specs = [by_name[name] for name in wanted]
        budget = QUICK_BUDGET if args.quick else FULL_BUDGET
        document = run_campaign(specs, budget, base_seed=args.seed, draws=args.draws)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output} ({document['elapsed_seconds']:.0f} s)")

    table = render_markdown_table(document)
    print()
    print(table)
    if args.write_docs:
        write_docs_block(DOCS_PAGE, table)
        print(f"\nupdated {DOCS_PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
