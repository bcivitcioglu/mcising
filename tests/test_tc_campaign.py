"""The Tc campaign: analysis helpers, committed results, and a slow rerun.

The campaign driver lives in ``scripts/tc_campaign.py`` — a script, not
part of the package (the library API is frozen before 1.0) — and is loaded
here by path. Three layers:

1. Fast, deterministic: the crossing and peak estimators recover synthetic
   truths and refuse to extrapolate.
2. Fast, data: the committed ``scripts/tc_campaign_results.json`` agrees
   with ``mcising.constants`` within the 2% gate, was produced at the full
   budget, and matches the table rendered in ``docs/advanced/physics.md``.
3. Slow, Monte Carlo: a quick-budget rerun on fresh RNG streams reproduces
   every lattice's Tc within 2% of literature and agrees with the committed
   headline values.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from mcising.constants import (
    TC_CUBIC_3D,
    TC_HONEYCOMB_2D,
    TC_SQUARE_2D,
    TC_TRIANGULAR_2D,
)

from tests._stats import DEFAULT_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tc_campaign.py"
RESULTS = REPO_ROOT / "scripts" / "tc_campaign_results.json"
DOCS_PAGE = REPO_ROOT / "docs" / "advanced" / "physics.md"

#: The roadmap gate: measured Tc within 2% of the literature value.
GATE_RELATIVE = 0.02
#: Floor on the rerun-vs-committed tolerance as a fraction of Tc. Four
#: combined sigmas is the nominal test; the floor absorbs the ~13% noise of
#: the jackknife errors themselves and keeps the nightly false-fail rate
#: negligible over a year of runs.
RERUN_FLOOR_RELATIVE = 0.0025
#: Fit-quality canary on the committed headline crossings: chi2/dof of the
#: quadratic fit (6 dof) beyond this would mean dishonest errors or an
#: inadequate local polynomial — the failure modes the campaign was
#: designed to expose, so the committed data must not carry them.
MAX_HEADLINE_CHI2_DOF = 3.0

CONSTANTS = {
    "square": TC_SQUARE_2D,
    "triangular": TC_TRIANGULAR_2D,
    "honeycomb": TC_HONEYCOMB_2D,
    "cubic": TC_CUBIC_3D,
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tc_campaign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tc_campaign"] = module
    spec.loader.exec_module(module)
    return module


tc = _load_script()


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(RESULTS.read_text(encoding="utf-8"))
    return document


def _synthetic_curves(
    grid: np.ndarray, crossing: float, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two curved 'cumulants' crossing at ``crossing``; the steeper one
    plays the larger lattice, as in the real data."""
    x = grid - crossing
    small = 0.61 - 1.0 * x - 8.0 * x**2 + rng.normal(0.0, noise, grid.size)
    large = 0.61 - 1.5 * x - 12.0 * x**2 + rng.normal(0.0, noise, grid.size)
    return small, large, np.full(grid.size, noise)


class TestBinderCrossing:
    def test_recovers_a_synthetic_crossing(self) -> None:
        rng = np.random.default_rng(1)
        grid = np.array(tc.temperature_grid(2.27))
        small, large, err = _synthetic_curves(grid, 2.27, 0.002, rng)
        c = tc.binder_crossing(
            grid, small, err, large, err, tc_ref=2.27, sizes=(48, 64)
        )
        assert c.tc is not None and c.error is not None
        assert abs(c.tc - 2.27) <= 4.0 * c.error, c
        assert 0.0005 < c.error < 0.01, c
        assert c.n_points == 9 and c.n_failed_draws == 0
        assert c.chi2_dof is not None and c.chi2_dof < 3.0

    def test_reference_only_selects_the_window(self) -> None:
        """Noise-free quadratics are fitted exactly, so moving the reference
        by one fine grid step (a different point set) leaves the root
        unchanged — tc_ref centres the data window and nothing else."""
        rng = np.random.default_rng(2)
        grid = np.array(tc.temperature_grid(2.27))
        small, large, err = _synthetic_curves(grid, 2.27, 0.0, rng)
        err = np.full(grid.size, 0.002)
        a = tc.binder_crossing(grid, small, err, large, err, tc_ref=2.27)
        b = tc.binder_crossing(grid, small, err, large, err, tc_ref=2.27 * 1.0025)
        assert a.tc is not None and b.tc is not None
        assert a.tc == pytest.approx(b.tc, abs=1e-9)
        assert a.tc == pytest.approx(2.27, abs=1e-9)

    def test_refuses_to_extrapolate(self) -> None:
        rng = np.random.default_rng(3)
        grid = np.array(tc.temperature_grid(2.27))
        # Crossing 2.6% above the reference: outside the +-1% window.
        small, large, err = _synthetic_curves(grid, 2.27 * 1.026, 0.001, rng)
        with pytest.raises(tc.CampaignError, match="need exactly one"):
            tc.binder_crossing(grid, small, err, large, err, tc_ref=2.27)

    def test_rejects_missing_error_bars(self) -> None:
        """Zero errors on both curves leave nothing to weight or bootstrap;
        a non-finite error is rejected outright."""
        rng = np.random.default_rng(4)
        grid = np.array(tc.temperature_grid(2.27))
        small, large, err = _synthetic_curves(grid, 2.27, 0.001, rng)
        zero = err * 0.0
        with pytest.raises(tc.CampaignError, match="error"):
            tc.binder_crossing(grid, small, zero, large, zero, tc_ref=2.27)
        bad = err.copy()
        bad[10] = math.nan
        with pytest.raises(tc.CampaignError, match="non-finite"):
            tc.binder_crossing(grid, small, bad, large, err, tc_ref=2.27)


class TestCvPeak:
    def test_recovers_a_synthetic_vertex(self) -> None:
        rng = np.random.default_rng(5)
        grid = np.array(tc.temperature_grid(2.27))
        cv = 2.0 - 50.0 * (grid - 2.28) ** 2 + rng.normal(0.0, 0.01, grid.size)
        p = tc.cv_peak(grid, cv, np.full(grid.size, 0.01), size=64)
        assert p.tc is not None and p.error is not None
        assert abs(p.tc - 2.28) <= 4.0 * p.error, p
        assert p.n_failed_draws == 0 and p.reason is None

    def test_unbracketed_peak_is_reported_not_extrapolated(self) -> None:
        grid = np.array(tc.temperature_grid(2.27))
        rising = np.linspace(1.0, 2.0, grid.size)
        p = tc.cv_peak(grid, rising, np.full(grid.size, 0.01), size=64)
        assert p.tc is None and p.error is None
        assert p.reason is not None and "bracket" in p.reason


class TestRendering:
    def test_docs_block_roundtrip(self, tmp_path: Path) -> None:
        page = tmp_path / "physics.md"
        page.write_text(
            f"intro\n{tc.DOCS_BEGIN}\nold\n{tc.DOCS_END}\noutro\n", encoding="utf-8"
        )
        tc.write_docs_block(page, "| new |")
        assert page.read_text(encoding="utf-8") == (
            f"intro\n{tc.DOCS_BEGIN}\n| new |\n{tc.DOCS_END}\noutro\n"
        )

    def test_docs_block_requires_markers(self, tmp_path: Path) -> None:
        page = tmp_path / "physics.md"
        page.write_text("no markers here\n", encoding="utf-8")
        with pytest.raises(tc.CampaignError, match="exactly one"):
            tc.write_docs_block(page, "| new |")


class TestCommittedResults:
    """The committed campaign: the roadmap gate and its bookkeeping."""

    def test_every_lattice_within_two_percent(self, committed: dict[str, Any]) -> None:
        assert set(committed["lattices"]) == set(CONSTANTS)
        for name, lat in committed["lattices"].items():
            measured = lat["tc_measured"]
            exact = CONSTANTS[name]
            deviation = abs(measured["value"] - exact)
            assert deviation <= GATE_RELATIVE * exact, (
                f"{name}: Tc = {measured['value']:.5f} ± {measured['error_stat']:.5f} "
                f"vs {exact:.5f} ({100 * deviation / exact:.2f}% > 2%)"
            )

    def test_uncertainties_are_quoted(self, committed: dict[str, Any]) -> None:
        for name, lat in committed["lattices"].items():
            measured = lat["tc_measured"]
            assert math.isfinite(measured["error_stat"]) and measured["error_stat"] > 0
            assert math.isfinite(measured["error_sys"]), name
            assert measured["method"] == "binder_crossing"
            assert measured["sizes"] == lat["sizes"][-2:]

    def test_reference_values_are_the_constants(
        self, committed: dict[str, Any]
    ) -> None:
        """A stale literature value in the JSON must not pass silently."""
        for name, lat in committed["lattices"].items():
            assert lat["tc_exact"] == pytest.approx(CONSTANTS[name], rel=1e-12), name

    def test_fits_are_honest(self, committed: dict[str, Any]) -> None:
        for name, lat in committed["lattices"].items():
            chi2 = lat["tc_measured"]["chi2_dof"]
            assert chi2 <= MAX_HEADLINE_CHI2_DOF, f"{name}: chi2/dof {chi2:.2f}"
            assert lat["stationarity_flags"] == 0, name

    def test_committed_run_used_the_full_budget(
        self, committed: dict[str, Any]
    ) -> None:
        budget = committed["budget"]
        assert budget["quick"] is False
        assert budget["n_sweeps"] == tc.FULL_BUDGET.n_sweeps
        assert budget["n_thermalization"] == tc.FULL_BUDGET.n_thermalization
        assert budget["algorithm"] == "swendsen_wang"
        assert budget["mode"] == "independent"
        for name, lat in committed["lattices"].items():
            spec = next(s for s in tc.LATTICES if s.name == name)
            assert lat["sizes"] == list(spec.sizes), name
            assert lat["temperatures"] == list(tc.temperature_grid(spec.tc_exact))

    def test_docs_table_matches_the_json(self, committed: dict[str, Any]) -> None:
        text = DOCS_PAGE.read_text(encoding="utf-8")
        assert text.count(tc.DOCS_BEGIN) == 1 and text.count(tc.DOCS_END) == 1
        block = text.split(tc.DOCS_BEGIN, 1)[1].split(tc.DOCS_END, 1)[0].strip("\n")
        assert block == tc.render_markdown_table(committed), (
            "docs table is stale: run "
            "`uv run python scripts/tc_campaign.py --from-json "
            "scripts/tc_campaign_results.json --write-docs`"
        )


@pytest.mark.slow
@pytest.mark.statistical
@pytest.mark.parametrize("seed", DEFAULT_SEEDS)
def test_quick_rerun_reproduces_the_committed_campaign(
    seed: int, committed: dict[str, Any]
) -> None:
    """A fresh quick-budget campaign lands within 2% of literature on every
    lattice and agrees with the committed headline values.

    Same sizes and grid as the committed run, so the finite-size systematics
    cancel and the comparison is purely statistical. ``+500`` moves every
    rerun off the committed campaign's RNG streams (seed 42 + 1000 k + i,
    i < 27): a rerun on the same stream would share its samples and prove
    nothing.
    """
    document = tc.run_campaign(
        tc.LATTICES, tc.QUICK_BUDGET, base_seed=seed + 500, log=lambda _: None
    )
    for name, lat in document["lattices"].items():
        exact = CONSTANTS[name]
        rerun = lat["tc_measured"]
        reference = committed["lattices"][name]["tc_measured"]
        deviation = abs(rerun["value"] - exact)
        assert deviation <= GATE_RELATIVE * exact, (
            f"{name} (seed={seed}): Tc = {rerun['value']:.5f} ± "
            f"{rerun['error_stat']:.5f} vs literature {exact:.5f} "
            f"({100 * deviation / exact:.2f}% > 2%)"
        )
        combined = math.hypot(rerun["error_stat"], reference["error_stat"])
        tolerance = max(4.0 * combined, RERUN_FLOOR_RELATIVE * exact)
        gap = abs(rerun["value"] - reference["value"])
        assert gap <= tolerance, (
            f"{name} (seed={seed}): rerun {rerun['value']:.5f} ± "
            f"{rerun['error_stat']:.5f} vs committed {reference['value']:.5f} ± "
            f"{reference['error_stat']:.5f}: gap {gap:.5f} > {tolerance:.5f} "
            f"({gap / combined:.1f} sigma)"
        )
