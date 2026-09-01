"""Golden replay: every execution path reproduces its committed fixture.

The fixture (``tests/data/golden_runs.json``) is captured by
``scripts/capture_golden.py`` and pins, bit for bit, the RNG streams and the
observable arithmetic of every run path — cooldown per algorithm and
lattice, independent, parallel tempering and adaptive. A refactor that keeps
the physics keeps this file green without regenerating the fixture (the
regeneration policy is in the script's docstring).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "capture_golden.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("capture_golden", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["capture_golden"] = module
    spec.loader.exec_module(module)
    return module


golden = _load_script()
CASE_NAMES = [case.name for case in golden.CASES]


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        golden.FIXTURE_PATH.read_text(encoding="utf-8")
    )
    return document


def test_fixture_covers_every_case_in_order(committed: dict[str, Any]) -> None:
    assert committed["schema"] == golden.SCHEMA_VERSION
    assert [case["name"] for case in committed["cases"]] == CASE_NAMES


def test_fixture_records_provenance(committed: dict[str, Any]) -> None:
    provenance = committed["provenance"]
    assert set(provenance) >= {
        "mcising_version",
        "git_commit",
        "python",
        "platform",
        "machine",
        "captured",
    }
    assert provenance["mcising_version"]


def test_compare_distinguishes_signed_zero_and_length() -> None:
    assert golden.compare({"e": [0.0]}, {"e": [0.0]}) == []
    assert golden.compare({"e": [-0.0]}, {"e": [0.0]}) != []
    assert golden.compare({"e": [1.0, 2.0]}, {"e": [1.0]}) != []
    assert golden.compare({"e": [1.0]}, {"e": [1.0], "x": 1}) != []


@pytest.mark.parametrize("name", CASE_NAMES)
def test_replay_is_bit_identical(name: str, committed: dict[str, Any]) -> None:
    case = next(c for c in golden.CASES if c.name == name)
    expected = next(c for c in committed["cases"] if c["name"] == name)
    diffs = golden.compare(golden.run_case(case), expected, name)
    assert not diffs, "\n".join(diffs[:10])
