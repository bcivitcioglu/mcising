"""P11: matplotlib is the optional ``plot`` extra.

``import mcising`` must never touch matplotlib, and the core simulation
workflow must run without it. conftest imports matplotlib into this
process for the plotting tests, so the guarantee is only testable in a
subprocess with an import blocker on the meta path.
"""

from __future__ import annotations

import subprocess
import sys

_BLOCKED_WORKFLOW = """
import sys


class _BlockMatplotlib:
    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib blocked by test")
        return None


sys.meta_path.insert(0, _BlockMatplotlib())

import mcising

assert "matplotlib" not in sys.modules
assert "mcising.plotting" not in sys.modules

# The core workflow works without matplotlib.
from mcising import LatticeConfig, Simulation, SimulationConfig

config = SimulationConfig(
    lattice=LatticeConfig(size=4),
    temperatures=(3.0,),
    n_sweeps=10,
    n_thermalization=5,
)
results = Simulation(config).run(show_progress=False)
assert 3.0 in results.energy

# Accessing a plotting export without matplotlib raises the friendly
# ImportError that names the extra.
try:
    mcising.plot_energy
except ImportError as exc:
    assert "mcising[plot]" in str(exc), str(exc)
else:
    raise AssertionError("plot_energy access should require matplotlib")

print("OK")
"""


class TestMatplotlibOptional:
    def test_workflow_runs_with_matplotlib_blocked(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", _BLOCKED_WORKFLOW],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().endswith("OK")

    def test_import_mcising_does_not_import_matplotlib(self) -> None:
        code = (
            "import sys, mcising; "
            "assert 'matplotlib' not in sys.modules; print('OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr

    def test_lazy_plotting_access_works_when_installed(self) -> None:
        # In-process: matplotlib IS installed in the dev env, so the
        # lazy attribute must resolve to the real function.
        import mcising

        assert callable(mcising.plot_energy)

    def test_unknown_attribute_still_raises(self) -> None:
        import mcising

        try:
            mcising.definitely_not_an_export
        except AttributeError as exc:
            assert "definitely_not_an_export" in str(exc)
        else:
            raise AssertionError("expected AttributeError")
