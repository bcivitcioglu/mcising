"""Every script in ``examples/`` runs to completion and writes its figure.

The canonical suite runs each example with ``--quick`` (seconds) so an API
drift fails fast on every push; the slow suite runs the full budget the
documentation shows, with the roadmap's five-minute bound as a timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((REPO_ROOT / "examples").glob("*.py"))
FULL_BUDGET_TIMEOUT_S = 300
MIN_FIGURE_BYTES = 10_000


def _run_example(
    script: Path, out_dir: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir), *extra],
        capture_output=True,
        text=True,
        env=env,
        timeout=FULL_BUDGET_TIMEOUT_S,
        check=False,
    )


def _assert_figure_written(
    script: Path, out_dir: Path, result: subprocess.CompletedProcess[str]
) -> None:
    assert result.returncode == 0, (
        f"{script.name} failed:\n{result.stdout}\n{result.stderr}"
    )
    figure = out_dir / f"{script.stem}.png"
    assert figure.is_file(), f"{script.name} wrote no {figure.name}:\n{result.stdout}"
    assert figure.stat().st_size > MIN_FIGURE_BYTES, f"{figure} is implausibly small"
    assert f"wrote {figure}" in result.stdout


def test_examples_are_present() -> None:
    names = {p.name for p in EXAMPLES}
    assert {
        "onsager_reproduction.py",
        "stripe_phase_diagram.py",
        "tc_binder_crossing.py",
    } <= names


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_quick_budget_writes_figure(script: Path, tmp_path: Path) -> None:
    result = _run_example(script, tmp_path, "--quick")
    _assert_figure_written(script, tmp_path, result)


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_full_budget_writes_figure(script: Path, tmp_path: Path) -> None:
    """The documented budget completes within the five-minute bound."""
    result = _run_example(script, tmp_path)
    _assert_figure_written(script, tmp_path, result)
