"""Tests for the Typer CLI."""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

import mcising
from mcising.config import ExecutionMode, LatticeConfig, SimulationConfig
from mcising.io import save_hdf5
from mcising.simulation import Simulation
from typer.testing import CliRunner

from mcising.cli import app  # isort: skip

runner = CliRunner()

#: Typer renders usage errors through rich, which colours the offending option
#: name. On a CI runner that styling splits a flag into several escape
#: sequences, so a test that looks for the flag in the output must ask for
#: plain text first.
PLAIN_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


class TestRunOptionValidation:
    """P11: enum-typed options give exit-2 usage errors, not tracebacks."""

    def test_bogus_lattice_exits_2_with_usage_error(self) -> None:
        result = runner.invoke(app, ["run", "--lattice", "bogus"])
        assert result.exit_code == 2
        assert "bogus" in result.output

    def test_bogus_algorithm_exits_2(self) -> None:
        result = runner.invoke(app, ["run", "--algorithm", "bogus"])
        assert result.exit_code == 2

    def test_bogus_mode_exits_2(self) -> None:
        result = runner.invoke(app, ["run", "--mode", "bogus"])
        assert result.exit_code == 2

    def test_run_help_shows_swap_interval(self) -> None:
        # The literal gate: --swap-interval appears in the rendered
        # help. rich emits ANSI-styled help when it detects CI (option
        # names split across style spans), so strip escapes first.
        result = runner.invoke(
            app,
            ["run", "--help"],
            env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"},
        )
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--swap-interval" in plain

    def test_run_declares_all_new_flags(self) -> None:
        # Environment-independent form of the same contract: rich wraps
        # long compound names (--store-configs/--no-store-configs)
        # unpredictably in rendered help, so the full flag set is
        # asserted on the click parameter declarations that generate it.
        from typer.main import get_command

        run_params = get_command(app).commands["run"].params
        opts: set[str] = set()
        for param in run_params:
            opts.update(getattr(param, "opts", []))
            opts.update(getattr(param, "secondary_opts", []))
        assert {
            "--swap-interval",
            "--c-window",
            "--tau-multiplier",
            "--min-therm",
            "--max-therm",
            "--store-configs",
            "--no-store-configs",
        } <= opts

    def test_swap_interval_plumbs_to_config(self, tmp_path: Path) -> None:
        out = tmp_path / "pt.h5"
        result = runner.invoke(
            app,
            [
                "run",
                "-L",
                "4",
                "-T",
                "3.0",
                "-T",
                "2.5",
                "--mode",
                "parallel_tempering",
                "--swap-interval",
                "2",
                "--interval",
                "4",
                "--sweeps",
                "8",
                "--therm",
                "4",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_incompatible_swap_interval_surfaces_config_error(self) -> None:
        # swap_interval must divide measurement_interval (B5); the flag
        # plumbing is proven by the validation firing.
        result = runner.invoke(
            app,
            [
                "run",
                "--mode",
                "parallel_tempering",
                "--swap-interval",
                "3",
                "--interval",
                "10",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, mcising.ConfigurationError)


class TestInfo:
    def test_info_prints_installed_version(self) -> None:
        # Gate: `mcising info` reports the pyproject version — i.e. the
        # installed distribution's, not a hardcoded constant (B12).
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert mcising.__version__ in result.stdout
        assert mcising.__version__ == importlib.metadata.version("mcising")


class TestSummaryJson:
    def test_summary_json_has_pt_diagnostics(self, tmp_path: Path) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(3.0, 2.0),
            n_sweeps=20,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "pt.h5"
        save_hdf5(results, path)

        result = runner.invoke(app, ["summary", str(path), "--json"])
        assert result.exit_code == 0
        record = json.loads(result.stdout)["parallel_tempering"]
        assert record["swap_attempted"] == [10]  # 20 rounds, pairs on even rounds
        assert record["temperatures"] == [2.0, 3.0]

        # The plain summary prints the exchange line.
        result = runner.invoke(app, ["summary", str(path)])
        assert result.exit_code == 0
        assert "Replica exchange" in result.stdout

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


class TestSummaryErrors:
    """P08: summary output quotes uncertainties (B10)."""

    def _saved_run(self, tmp_path: Path, n_sweeps: int = 400) -> Path:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.0),
            n_sweeps=n_sweeps,
            measurement_interval=10,
        )
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "results.h5"
        save_hdf5(results, path)
        return path

    def test_summary_json_has_error_fields(self, tmp_path: Path) -> None:
        path = self._saved_run(tmp_path)
        result = runner.invoke(app, ["summary", str(path), "--json"])
        assert result.exit_code == 0
        rows = json.loads(result.stdout)["results"]
        for row in rows:
            assert row["E_err"] > 0.0
            assert row["Cv_err"] > 0.0
            assert row["chi_err"] > 0.0
            assert "U4" in row
            assert row["tau_int"] >= 0.5

    def test_summary_json_omits_nan_errors(self, tmp_path: Path) -> None:
        # 2 samples: jackknife errors are NaN by policy; strict JSON has
        # no NaN, so those keys must be absent (and the output parseable).
        path = self._saved_run(tmp_path, n_sweeps=20)
        result = runner.invoke(app, ["summary", str(path), "--json"])
        assert result.exit_code == 0
        rows = json.loads(result.stdout)["results"]
        for row in rows:
            assert "Cv" in row
            assert "Cv_err" not in row

    def test_summary_csv_header_matches_rows(self, tmp_path: Path) -> None:
        path = self._saved_run(tmp_path)
        result = runner.invoke(app, ["summary", str(path), "--csv"])
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().splitlines() if "," in ln]
        header = lines[0].split(",")
        assert header[:4] == ["T", "E_mean", "E_err", "E_std"]
        assert "Cv_err" in header
        assert "U4" in header
        for line in lines[1:]:
            assert len(line.split(",")) == len(header)


class TestWangLandauCommand:
    """`mcising wang-landau` and the Wang-Landau-aware `summary`."""

    def test_runs_saves_and_summarizes(self, tmp_path: Path) -> None:
        h5 = tmp_path / "dos.h5"
        js = tmp_path / "dos.json"
        result = runner.invoke(
            app,
            [
                "wang-landau",
                "-L",
                "4",
                "--log-f-final",
                "1e-3",
                "--check-interval",
                "100",
                "--production-sweeps",
                "200",
                "--walkers",
                "2",
                "-T",
                "2.0",
                "-T",
                "3.0",
                "-o",
                str(h5),
                "--json",
                str(js),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Wang-Landau" in result.output and "Saved HDF5" in result.output
        assert h5.exists() and js.exists()
        data = json.loads(js.read_text())
        assert data["kind"] == "wang_landau"
        assert set(data["results"]) == {"2.000000", "3.000000"}

        table = runner.invoke(app, ["summary", str(h5), "-T", "2.269"])
        assert table.exit_code == 0, table.output
        assert "Reweighted" in table.output and "converged" in table.output

        as_json = runner.invoke(app, ["summary", str(h5), "--json", "-T", "2.0"])
        assert as_json.exit_code == 0, as_json.output
        payload = json.loads(as_json.output)
        assert payload["kind"] == "wang_landau"
        assert "energy" in payload["results"]["2.000000"]

        as_csv = runner.invoke(
            app, ["summary", str(h5), "--csv", "-T", "2.0", "-T", "3.0"]
        )
        assert as_csv.exit_code == 0, as_csv.output
        lines = [line for line in as_csv.output.splitlines() if line.strip()]
        assert lines[0].startswith("T,E,E_err,Cv,Cv_err")
        assert len(lines) == 3

    def test_windowed_cubic_run_and_tip(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "wang-landau",
                "-L",
                "4",
                "--lattice",
                "cubic",
                "--j2",
                "-0.5",
                "--energy-window",
                "-1.7:-0.5",
                "--log-f-final",
                "1e-2",
                "--check-interval",
                "50",
                "--production-sweeps",
                "50",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Tip: use -o" in result.output

    def test_bad_energy_window_exits_2(self) -> None:
        for value in ("1.0", "a:b", "0.5:-0.5"):
            result = runner.invoke(
                app, ["wang-landau", "--energy-window", value], env=PLAIN_ENV
            )
            assert result.exit_code == 2, value
            assert "energy-window" in result.output

    def test_canonical_summary_ignores_temperatures_and_info_lists_schemas(
        self, tmp_path: Path
    ) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4), temperatures=(2.0,), n_sweeps=20
        )
        path = tmp_path / "sim.h5"
        save_hdf5(Simulation(config).run(show_progress=False), path)
        result = runner.invoke(app, ["summary", str(path), "-T", "3.0"])
        assert result.exit_code == 0, result.output
        info = runner.invoke(app, ["info"])
        assert info.exit_code == 0
        assert "Wang-Landau HDF5 schema" in info.output
        docs = runner.invoke(app, ["docs", "cli"])
        assert "wang-landau" in docs.output
        algorithms = runner.invoke(app, ["docs", "algorithms"])
        assert "wang_landau" in algorithms.output
