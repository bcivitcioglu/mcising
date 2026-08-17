"""Tests for the Typer CLI."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import mcising
from mcising.config import LatticeConfig, SimulationConfig
from mcising.io import save_hdf5
from mcising.simulation import Simulation
from typer.testing import CliRunner

from mcising.cli import app  # isort: skip

runner = CliRunner()


class TestInfo:
    def test_info_prints_installed_version(self) -> None:
        # Gate: `mcising info` reports the pyproject version — i.e. the
        # installed distribution's, not a hardcoded constant (B12).
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert mcising.__version__ in result.stdout
        assert mcising.__version__ == importlib.metadata.version("mcising")


class TestSummaryJson:
    def test_summary_json_has_provenance(self, tmp_path: Path) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(3.0, 2.0),
            n_sweeps=20,
            measurement_interval=10,
        )
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "results.h5"
        save_hdf5(results, path)

        result = runner.invoke(app, ["summary", str(path), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["version"] == mcising.__version__
        assert payload["seed"] == 42
        assert payload["mode"] == "cooldown"
        assert payload["algorithm"] == "metropolis"
        rows = payload["results"]
        assert len(rows) == 2
        assert {row["T"] for row in rows} == {2.0, 3.0}
