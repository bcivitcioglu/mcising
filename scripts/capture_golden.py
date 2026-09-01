#!/usr/bin/env python3
"""Golden fixtures: a bit-exact replay oracle for every execution path.

Each case in ``CASES`` is a small, fixed-seed ``SimulationConfig`` covering
one execution path (cooldown per algorithm and lattice, independent,
parallel tempering, adaptive) and one coupling family (J1 only, J1-J2-h,
J1-J3, antiferromagnetic). ``run_case`` replays a case and records every
number the run produced: the energy and magnetization series verbatim (JSON
floats round-trip float64 exactly), correlation data when enabled, the
adaptive diagnostics, and SHA-256 digests of the stored configurations, the
final spin state and the final RNG state (cooldown paths only — the parallel
runners own their replicas). ``compare`` reports every field that differs,
bit for bit (floats are compared through ``float.hex`` so ``-0.0`` and
``0.0`` are distinct).

``tests/test_golden.py`` replays every case against the committed
``tests/data/golden_runs.json``. The fixture pins the RNG streams and the
observable arithmetic of the whole library: a refactor that keeps the
physics must keep this file green without touching it.

Regeneration policy: rewrite the fixture (``--write``) only for an
*intentional* change of the random-number consumption order or of the
observable arithmetic, in the same commit as a CHANGELOG entry that says so
and, per the P15 rule, a 3-sigma statistical agreement check of the affected
path against the previous release.

Usage:
    uv run python scripts/capture_golden.py --check   # replay and compare
    uv run python scripts/capture_golden.py --write   # regenerate the fixture
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mcising
import numpy as np
from mcising import Simulation, SimulationConfig
from mcising._provenance import git_commit
from mcising.config import ExecutionMode

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
FIXTURE_PATH: Final = REPO_ROOT / "tests" / "data" / "golden_runs.json"
#: Bump when the *record layout* changes (not when values change).
SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class GoldenCase:
    """One replayable configuration.

    Parameters
    ----------
    name : str
        Stable identifier; also the test id.
    config : dict[str, Any]
        Input for :meth:`SimulationConfig.from_dict` (kept as a plain dict so
        the fixture records exactly what was run).
    note : str
        Which path or coupling family the case pins.
    """

    name: str
    config: dict[str, Any]
    note: str


def _square(size: int, **couplings: float) -> dict[str, Any]:
    return {"lattice_type": "square", "size": size, **couplings}


CASES: Final[tuple[GoldenCase, ...]] = (
    GoldenCase(
        "metropolis_square_cooldown",
        {
            "lattice": _square(16, j1=1.0),
            "algorithm": "metropolis",
            "temperatures": [3.0, 2.269, 1.5],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
        },
        "default path: Metropolis cooldown, J1 only, configurations stored",
    ),
    GoldenCase(
        "metropolis_square_j1j2h_cooldown",
        {
            "lattice": _square(12, j1=1.0, j2=-0.3, h=0.1),
            "algorithm": "metropolis",
            "temperatures": [2.0, 1.0],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
        },
        "J1-J2-h couplings (non-dyadic: exercises the serial energy sum)",
    ),
    GoldenCase(
        "metropolis_square_j3_cooldown",
        {
            "lattice": _square(12, j1=1.0, j3=0.25),
            "algorithm": "metropolis",
            "temperatures": [2.5],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
        },
        "J1-J3 couplings (dyadic: exercises the integer-shell energy sum)",
    ),
    GoldenCase(
        "metropolis_triangular_afm_cooldown",
        {
            "lattice": {"lattice_type": "triangular", "size": 12, "j1": -1.0},
            "algorithm": "metropolis",
            "temperatures": [2.0, 0.5],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
            "store_configs": False,
        },
        "antiferromagnetic triangular (B1 regression), configurations off",
    ),
    GoldenCase(
        "metropolis_honeycomb_cooldown",
        {
            "lattice": {"lattice_type": "honeycomb", "size": 8, "j1": 1.0},
            "algorithm": "metropolis",
            "temperatures": [2.0, 1.5187, 1.0],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
        },
        "honeycomb (three-index shape [L, L, 2])",
    ),
    GoldenCase(
        "metropolis_cubic_cooldown",
        {
            "lattice": {"lattice_type": "cubic", "size": 6, "j1": 1.0},
            "algorithm": "metropolis",
            "temperatures": [5.0, 4.5115, 4.0],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
        },
        "cubic (3D, z=6)",
    ),
    GoldenCase(
        "wolff_chain_corr_cooldown",
        {
            "lattice": {"lattice_type": "chain", "size": 64, "j1": 1.0},
            "algorithm": "wolff",
            "temperatures": [1.2, 0.8],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
            "compute_correlation": True,
        },
        "Wolff on the chain with correlation data",
    ),
    GoldenCase(
        "wolff_square_corr_cooldown",
        {
            "lattice": _square(16, j1=1.0),
            "algorithm": "wolff",
            "temperatures": [2.5, 2.269],
            "n_sweeps": 100,
            "n_thermalization": 100,
            "measurement_interval": 10,
            "compute_correlation": True,
        },
        "Wolff cooldown; correlation length at every measurement (10 per T)",
    ),
    GoldenCase(
        "sw_square_independent_campaign",
        {
            "lattice": _square(16, j1=1.0),
            "algorithm": "swendsen_wang",
            "mode": "independent",
            "temperatures": [2.4, 2.32, 2.269, 2.22, 2.15],
            "n_sweeps": 100,
            "n_thermalization": 50,
            "measurement_interval": 10,
            "store_configs": False,
            "compute_correlation": True,
        },
        "Tc-campaign grid shape: SW, independent mode (pins the per-temperature "
        "seed streams behind scripts/tc_campaign_results.json)",
    ),
    GoldenCase(
        "metropolis_square_pt",
        {
            "lattice": _square(8, j1=1.0),
            "algorithm": "metropolis",
            "mode": "parallel_tempering",
            "temperatures": [3.0, 2.5, 2.269, 2.0],
            "n_sweeps": 200,
            "n_thermalization": 100,
            "measurement_interval": 10,
            "swap_interval": 2,
        },
        "parallel tempering ladder (replica seeds + swap RNG)",
    ),
    GoldenCase(
        "metropolis_square_adaptive",
        {
            "lattice": _square(16, j1=1.0),
            "algorithm": "metropolis",
            "temperatures": [3.0, 2.269],
            "compute_correlation": True,
            "adaptive": {"enabled": True},
        },
        "adaptive thermalization (anneal + extend + production_sweeps) with the "
        "single end-of-production correlation snapshot and every diagnostic",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).tolist()]


def run_case(case: GoldenCase) -> dict[str, Any]:
    """Replay one case and return its complete record."""
    config = SimulationConfig.from_dict(case.config)
    sim = Simulation(config)
    results = sim.run(show_progress=False)

    entries: list[dict[str, Any]] = []
    for temp in results.temperatures:
        entry: dict[str, Any] = {
            "temperature": float(temp),
            "energies": _floats(results.energy[temp]),
            "magnetizations": _floats(results.magnetization[temp]),
            "n_cluster_flips": int(results.n_cluster_flips[temp]),
        }
        if temp in results.configurations:
            configs = np.ascontiguousarray(results.configurations[temp])
            entry["configurations_shape"] = list(configs.shape)
            entry["configurations_dtype"] = str(configs.dtype)
            entry["configurations_sha256"] = _sha256(configs.tobytes())
        if (
            results.correlation_length is not None
            and results.correlation_function is not None
            and temp in results.correlation_length
        ):
            distances, values = results.correlation_function[temp]
            entry["correlation_length"] = _floats(results.correlation_length[temp])
            entry["correlation_distances"] = _floats(distances)
            entry["correlation_function"] = _floats(values)
        if results.adaptive_diagnostics is not None:
            entry["adaptive_diagnostics"] = dataclasses.asdict(
                results.adaptive_diagnostics[temp]
            )
        entries.append(entry)

    record: dict[str, Any] = {
        "name": case.name,
        "note": case.note,
        "config": case.config,
        "temperatures": [float(t) for t in results.temperatures],
        "per_temperature": entries,
    }
    if config.mode == ExecutionMode.COOLDOWN:
        # The parallel runners advance their own replicas; only the cooldown
        # paths leave the Simulation's core in the final state.
        record["final_spins_sha256"] = _sha256(
            np.ascontiguousarray(sim.spins).tobytes()
        )
        record["final_rng_state_sha256"] = _sha256(
            bytes(sim._core.get_rng_state())  # noqa: SLF001 - fixture oracle
        )
    return record


def _describe(value: Any) -> str:
    if isinstance(value, float):
        return f"{value!r} ({value.hex()})"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return repr(value)


def compare(actual: Any, expected: Any, path: str = "") -> list[str]:
    """Return one message per field that differs, bit for bit."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a mapping, got {type(actual).__name__}"]
        diffs: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in actual:
                diffs.append(f"{sub}: missing from the replay")
            elif key not in expected:
                diffs.append(f"{sub}: not in the fixture")
            else:
                diffs.extend(compare(actual[key], expected[key], sub))
        return diffs
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected a list, got {type(actual).__name__}"]
        if len(actual) != len(expected):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        diffs = []
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            diffs.extend(compare(a, e, f"{path}[{i}]"))
        return diffs
    if isinstance(expected, float) or isinstance(actual, float):
        a = float(actual)
        e = float(expected)
        same = a.hex() == e.hex() or (math.isnan(a) and math.isnan(e))
        return [] if same else [f"{path}: {_describe(a)} != {_describe(e)}"]
    if actual != expected:
        return [f"{path}: {_describe(actual)} != {_describe(expected)}"]
    return []


def capture() -> dict[str, Any]:
    """Replay every case and assemble the fixture document."""
    return {
        "schema": SCHEMA_VERSION,
        "provenance": {
            "mcising_version": mcising.__version__,
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "captured": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        },
        "cases": [run_case(case) for case in CASES],
    }


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Read the committed fixture."""
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def check(path: Path = FIXTURE_PATH) -> list[str]:
    """Replay every case against ``path``; return all differences."""
    committed = load_fixture(path)
    by_name = {c["name"]: c for c in committed["cases"]}
    diffs: list[str] = []
    if list(by_name) != [c.name for c in CASES]:
        diffs.append(f"case list differs: fixture {list(by_name)}")
    for case in CASES:
        if case.name not in by_name:
            continue
        diffs.extend(compare(run_case(case), by_name[case.name], case.name))
    return diffs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--write", action="store_true", help="regenerate the fixture (see policy)"
    )
    action.add_argument(
        "--check", action="store_true", help="replay and compare bit for bit"
    )
    parser.add_argument(
        "--output", type=Path, default=FIXTURE_PATH, help="fixture path"
    )
    args = parser.parse_args(argv)

    if args.write:
        document = capture()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(document['cases'])} cases to {args.output}")
        return 0

    diffs = check(args.output)
    if diffs:
        print("\n".join(diffs[:50]))
        print(f"{len(diffs)} difference(s)")
        return 1
    print(f"all {len(CASES)} cases replay bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
