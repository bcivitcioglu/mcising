"""P12: CliRunner coverage for every non-`run` CLI subcommand.

Every subcommand gets at least one success and one failure case. Output
assertions strip ANSI first: rich emits styled output when it detects CI
(the P11 lesson), so raw-substring asserts are environment-dependent.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest
from mcising.benchmarks import BenchmarkResult
from mcising.config import LatticeConfig, SimulationConfig
from mcising.io import save_hdf5
from mcising.simulation import Simulation
from typer.testing import CliRunner

from mcising.cli import app  # isort: skip

runner = CliRunner()

# Deterministic rendering for output assertions (see test_cli.py).
PLAIN_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def _plain(output: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def _saved_results(path: Path, seed: int = 42) -> Path:
    config = SimulationConfig(
        lattice=LatticeConfig(size=4),
        temperatures=(3.0, 2.0),
        n_sweeps=40,
        measurement_interval=10,
        compute_correlation=True,
        seed=seed,
    )
    results = Simulation(config).run(show_progress=False)
    save_hdf5(results, path)
    return path


@pytest.fixture(scope="module")
def results_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One tiny real HDF5 file shared by all summary/plot/export tests."""
    return _saved_results(tmp_path_factory.mktemp("cli_data") / "results.h5")


@pytest.fixture(scope="module")
def results_file_b(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second file (different seed) for multi-file overlay plots."""
    return _saved_results(
        tmp_path_factory.mktemp("cli_data_b") / "results_b.h5", seed=43
    )


class TestInfo:
    """Success case lives in test_cli.py::TestInfo."""

    def test_info_rejects_unknown_flag(self) -> None:
        result = runner.invoke(app, ["info", "--bogus"])
        assert result.exit_code == 2


class TestDocs:
    @pytest.mark.parametrize(
        ("topic", "expected"),
        [
            ("lattices", "honeycomb"),
            ("algorithms", "swendsen_wang"),
            ("couplings", "COUPLING SUPPORT"),
            ("modes", "parallel_tempering"),
            ("cli", "mcising CLI REFERENCE"),
        ],
    )
    def test_docs_topics_print_reference(self, topic: str, expected: str) -> None:
        result = runner.invoke(app, ["docs", topic])
        assert result.exit_code == 0, result.output
        assert expected in result.output

    def test_docs_bare_shows_full_reference(self) -> None:
        result = runner.invoke(app, ["docs"])
        assert result.exit_code == 0, result.output
        assert "mcising CLI REFERENCE" in result.output

    def test_docs_unknown_topic_exits_2(self) -> None:
        result = runner.invoke(app, ["docs", "bogus"])
        assert result.exit_code == 2


class TestSummary:
    """--json/--csv paths live in test_cli.py; this covers the rest."""

    def test_summary_rich_table_lists_temperatures(
        self, results_file: Path
    ) -> None:
        result = runner.invoke(app, ["summary", str(results_file)], env=PLAIN_ENV)
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "Simulation Results" in plain
        assert "3.0000" in plain
        assert "2.0000" in plain

    def test_summary_json_wins_over_csv(self, results_file: Path) -> None:
        # Pinned precedence: both flags given, JSON is emitted.
        result = runner.invoke(
            app, ["summary", str(results_file), "--json", "--csv"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert len(payload["results"]) == 2

    def test_summary_missing_file_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["summary", str(tmp_path / "nope.h5")])
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)


class TestPlotVsTemperature:
    """The four observable-vs-T plot subcommands."""

    def _assert_writes_png(self, cmd: str, source: Path, tmp_path: Path) -> None:
        out = tmp_path / f"{cmd}.png"
        result = runner.invoke(app, ["plot", cmd, str(source), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_energy_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        self._assert_writes_png("energy", results_file, tmp_path)

    def test_plot_magnetization_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        self._assert_writes_png("magnetization", results_file, tmp_path)

    def test_plot_specific_heat_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        self._assert_writes_png("specific-heat", results_file, tmp_path)

    def test_plot_susceptibility_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        self._assert_writes_png("susceptibility", results_file, tmp_path)

    def test_plot_energy_multi_file_overlay(
        self, results_file: Path, results_file_b: Path, tmp_path: Path
    ) -> None:
        # Two files exercise the list-source branch of the command body.
        out = tmp_path / "overlay.png"
        result = runner.invoke(
            app,
            ["plot", "energy", str(results_file), str(results_file_b), "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_plot_energy_missing_output_flag_exits_2(
        self, results_file: Path
    ) -> None:
        result = runner.invoke(app, ["plot", "energy", str(results_file)])
        assert result.exit_code == 2

    def test_plot_energy_missing_input_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "o.png"
        result = runner.invoke(
            app, ["plot", "energy", str(tmp_path / "nope.h5"), "-o", str(out)]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, OSError)

    def test_plot_bare_exits_2(self) -> None:
        result = runner.invoke(app, ["plot"])
        assert result.exit_code == 2


class TestPlotLattice:
    def test_plot_lattice_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "lattice.png"
        result = runner.invoke(
            app,
            ["plot", "lattice", str(results_file), "-T", "3.0", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_plot_lattice_single_config_index(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "lattice0.png"
        result = runner.invoke(
            app,
            [
                "plot",
                "lattice",
                str(results_file),
                "-T",
                "3.0",
                "--n",
                "0",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_plot_lattice_index_out_of_range_fails(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "plot",
                "lattice",
                str(results_file),
                "-T",
                "3.0",
                "--n",
                "99",
                "-o",
                str(tmp_path / "o.png"),
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_plot_lattice_missing_temperature_exits_2(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["plot", "lattice", str(results_file), "-o", str(tmp_path / "o.png")]
        )
        assert result.exit_code == 2


class TestPlotTimeseries:
    def test_plot_timeseries_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "ts.png"
        result = runner.invoke(
            app,
            ["plot", "timeseries", str(results_file), "-T", "2.0", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_plot_timeseries_unknown_temperature_fails(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "plot",
                "timeseries",
                str(results_file),
                "-T",
                "9.9",
                "-o",
                str(tmp_path / "o.png"),
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)


class TestPlotHistogram:
    def test_plot_histogram_writes_png(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "hist.png"
        result = runner.invoke(
            app,
            ["plot", "histogram", str(results_file), "-T", "3.0", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_plot_histogram_unknown_temperature_fails(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "plot",
                "histogram",
                str(results_file),
                "-T",
                "9.9",
                "-o",
                str(tmp_path / "o.png"),
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)


class TestExport:
    def test_export_writes_zip_of_pngs(
        self, results_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "configs.zip"
        result = runner.invoke(app, ["export", str(results_file), str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert names
        assert all(name.endswith(".png") for name in names)

    def test_export_missing_input_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["export", str(tmp_path / "nope.h5"), str(tmp_path / "o.zip")]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, OSError)

    def test_export_without_args_exits_2(self) -> None:
        result = runner.invoke(app, ["export"])
        assert result.exit_code == 2


class TestBenchmark:
    def test_benchmark_small_run_prints_tables(self) -> None:
        result = runner.invoke(
            app, ["benchmark", "-L", "4", "--sweeps", "5"], env=PLAIN_ENV
        )
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "Metropolis Performance" in plain
        assert "Cluster Algorithms" in plain
        assert "Coupling Strategies" in plain

    def test_benchmark_scaling_runs_all_sizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The scaling path hardcodes sizes up to 256; a real run belongs
        # in benchmarks, not the test suite, so the bench call is faked.
        calls: list[int] = []

        def fake_bench(size: int, sweeps: int, seed: int) -> BenchmarkResult:
            calls.append(size)
            return BenchmarkResult(
                name="fake",
                lattice_size=size,
                n_sweeps=sweeps,
                elapsed=0.001,
                energy=-1.0,
                magnetization=0.5,
                num_sites=size * size,
            )

        monkeypatch.setattr("mcising.benchmarks.bench_mcising", fake_bench)
        result = runner.invoke(app, ["benchmark", "--scaling"], env=PLAIN_ENV)
        assert result.exit_code == 0, result.output
        assert calls == [8, 16, 32, 64, 128, 256]
        assert "256" in _plain(result.output)

    def test_benchmark_invalid_size_fails(self) -> None:
        result = runner.invoke(app, ["benchmark", "-L", "0", "--sweeps", "1"])
        assert result.exit_code != 0
        assert result.exception is not None
