"""P12: CliRunner coverage for the `run` subcommand's flag matrix.

Covers --T-range parsing, checkpoint/resume combinations, mode/algorithm
constraint surfacing, adaptive knobs, and output-path branches. Output
assertions strip ANSI (the P11 lesson); every invocation uses the tiny
fast recipe so the whole module stays sub-second per test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import mcising
import pytest
from typer.testing import CliRunner

from mcising.cli import app  # isort: skip

runner = CliRunner()

PLAIN_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}
FAST = ["-L", "4", "--sweeps", "8", "--therm", "4", "--interval", "4"]


def _plain(output: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def _groups(path: Path) -> set[str]:
    with h5py.File(path, "r") as f:
        return {k for k in f if k.startswith("T=")}


class TestRunTRange:
    def test_run_t_range_generates_scan(self, tmp_path: Path) -> None:
        out = tmp_path / "scan.h5"
        result = runner.invoke(
            app, ["run", *FAST, "--T-range", "3.0:2.0:0.5", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert _groups(out) == {"T=3.000000", "T=2.500000", "T=2.000000"}

    def test_run_t_range_ascending_scan(self, tmp_path: Path) -> None:
        out = tmp_path / "up.h5"
        result = runner.invoke(
            app, ["run", *FAST, "--T-range", "2.0:3.0:0.5", "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert _groups(out) == {"T=2.000000", "T=2.500000", "T=3.000000"}

    @pytest.mark.parametrize(
        "bad_range",
        [
            "3.0:2.0",  # wrong part count
            "a:b:c",  # non-numeric
            "3.0:2.0:0",  # step must be positive
            "0:2.0:0.5",  # bounds must be positive
        ],
    )
    def test_run_t_range_invalid_exits_2(self, bad_range: str) -> None:
        result = runner.invoke(app, ["run", *FAST, "--T-range", bad_range])
        assert result.exit_code == 2, result.output

    def test_run_t_and_t_range_together_exits_2(self) -> None:
        result = runner.invoke(
            app, ["run", *FAST, "-T", "2.5", "--T-range", "3.0:2.0:0.5"]
        )
        assert result.exit_code == 2


class TestRunCheckpointResume:
    def test_run_checkpoint_writes_groups(self, tmp_path: Path) -> None:
        ck = tmp_path / "ck.h5"
        result = runner.invoke(
            app,
            ["run", *FAST, "-T", "3.0", "-T", "2.0", "--checkpoint", str(ck)],
            env=PLAIN_ENV,
        )
        assert result.exit_code == 0, result.output
        assert _groups(ck) == {"T=3.000000", "T=2.000000"}

    def test_run_resume_extends_completed_scan(self, tmp_path: Path) -> None:
        ck = tmp_path / "ck.h5"
        first = runner.invoke(app, ["run", *FAST, "-T", "3.0", "--checkpoint", str(ck)])
        assert first.exit_code == 0, first.output
        second = runner.invoke(
            app,
            [
                "run",
                *FAST,
                "-T",
                "3.0",
                "-T",
                "2.0",
                "--checkpoint",
                str(ck),
                "--resume",
            ],
        )
        assert second.exit_code == 0, second.output
        assert _groups(ck) == {"T=3.000000", "T=2.000000"}

    def test_run_existing_checkpoint_without_resume_fails(
        self, tmp_path: Path
    ) -> None:
        ck = tmp_path / "ck.h5"
        first = runner.invoke(app, ["run", *FAST, "-T", "3.0", "--checkpoint", str(ck)])
        assert first.exit_code == 0, first.output
        second = runner.invoke(
            app, ["run", *FAST, "-T", "3.0", "--checkpoint", str(ck)]
        )
        assert second.exit_code != 0
        assert isinstance(second.exception, mcising.ConfigurationError)

    def test_run_resume_mismatched_config_fails(self, tmp_path: Path) -> None:
        ck = tmp_path / "ck.h5"
        first = runner.invoke(app, ["run", *FAST, "-T", "3.0", "--checkpoint", str(ck)])
        assert first.exit_code == 0, first.output
        args = ["run", *FAST, "-T", "3.0", "--j1=0.5"]
        second = runner.invoke(
            app, [*args, "--checkpoint", str(ck), "--resume"]
        )
        assert second.exit_code != 0
        assert isinstance(second.exception, mcising.ConfigurationError)

    def test_run_resume_without_checkpoint_exits_2(self) -> None:
        # #45: previously silently ignored (fresh run, exit 0).
        result = runner.invoke(app, ["run", *FAST, "-T", "3.0", "--resume"])
        assert result.exit_code == 2

    def test_run_checkpoint_interval_plumbs(self, tmp_path: Path) -> None:
        ck = tmp_path / "ck.h5"
        result = runner.invoke(
            app,
            [
                "run",
                *FAST,
                "-T",
                "3.0",
                "-T",
                "2.5",
                "-T",
                "2.0",
                "--checkpoint",
                str(ck),
                "--checkpoint-interval",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert _groups(ck) == {"T=3.000000", "T=2.500000", "T=2.000000"}

    def test_run_output_same_as_checkpoint_skips_resave(
        self, tmp_path: Path
    ) -> None:
        ck = tmp_path / "ck.h5"
        result = runner.invoke(
            app,
            ["run", *FAST, "-T", "3.0", "--checkpoint", str(ck), "-o", str(ck)],
            env=PLAIN_ENV,
        )
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "Checkpoint:" in plain
        assert "Saved HDF5" not in plain


class TestRunModeAlgorithmConstraints:
    def test_run_wolff_antiferromagnetic_j1_fails(self) -> None:
        result = runner.invoke(
            app, ["run", *FAST, "-T", "3.0", "--algorithm", "wolff", "--j1=-1.0"]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, mcising.ConfigurationError)

    def test_run_wolff_with_j2_fails(self) -> None:
        result = runner.invoke(
            app, ["run", *FAST, "-T", "3.0", "--algorithm", "wolff", "--j2", "0.5"]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, mcising.ConfigurationError)

    def test_run_swendsen_wang_succeeds(self) -> None:
        result = runner.invoke(
            app, ["run", *FAST, "-T", "3.0", "--algorithm", "swendsen_wang"]
        )
        assert result.exit_code == 0, result.output

    def test_run_mode_independent_succeeds(self, tmp_path: Path) -> None:
        out = tmp_path / "indep.h5"
        args = ["run", *FAST, "-T", "3.0", "-T", "2.0", "--mode", "independent"]
        result = runner.invoke(app, [*args, "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert _groups(out) == {"T=3.000000", "T=2.000000"}

    def test_run_odd_triangular_size_fails(self) -> None:
        result = runner.invoke(
            app, ["run", "-L", "5", "--lattice", "triangular", "-T", "3.0"]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, mcising.ConfigurationError)


class TestRunOutputBranches:
    def test_run_json_summary_written(self, tmp_path: Path) -> None:
        out = tmp_path / "summary.json"
        result = runner.invoke(
            app, ["run", *FAST, "-T", "3.0", "--json", str(out)]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out.read_text())
        assert "3.000000" in payload["results"]

    def test_run_tip_shown_without_output_paths(self) -> None:
        result = runner.invoke(app, ["run", *FAST, "-T", "3.0"], env=PLAIN_ENV)
        assert result.exit_code == 0, result.output
        assert "Tip:" in _plain(result.output)

    def test_run_no_store_configs_omits_datasets(self, tmp_path: Path) -> None:
        out = tmp_path / "lean.h5"
        result = runner.invoke(
            app,
            ["run", *FAST, "-T", "3.0", "--no-store-configs", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        with h5py.File(out, "r") as f:
            assert "energy" in f["T=3.000000"]
            assert "configurations" not in f["T=3.000000"]

    def test_run_seed_and_couplings_recorded(self, tmp_path: Path) -> None:
        out = tmp_path / "prov.h5"
        result = runner.invoke(
            app,
            [
                "run",
                *FAST,
                "-T",
                "3.0",
                "--seed",
                "7",
                "--j2",
                "0.25",
                "--j3",
                "0.125",
                "--h",
                "0.5",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        with h5py.File(out, "r") as f:
            attrs = f["metadata"].attrs
            config = json.loads(attrs["config_json"])
            assert int(attrs["seed"]) == 7
            assert config["seed"] == 7
            assert config["lattice"]["j2"] == 0.25
            assert config["lattice"]["j3"] == 0.125
            assert config["lattice"]["h"] == 0.5


class TestRunConfigPanel:
    def test_run_lattice_label_reports_actual_type(self) -> None:
        # #46: the config panel used to hardcode "square".
        args = ["run", "-L", "4", "--lattice", "cubic", "-T", "3.0"]
        result = runner.invoke(
            app,
            [*args, "--sweeps", "4", "--therm", "2", "--interval", "2"],
            env=PLAIN_ENV,
        )
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "cubic" in plain
        assert "square" not in plain

    def test_run_help_does_not_claim_2d(self) -> None:
        # #47: 1D/3D lattices are supported.
        result = runner.invoke(app, ["run", "--help"], env=PLAIN_ENV)
        assert result.exit_code == 0
        assert "2D" not in _plain(result.output)

    def test_run_adaptive_knobs_render_in_panel(self) -> None:
        result = runner.invoke(
            app,
            [
                "run",
                *FAST,
                "-T",
                "3.0",
                "--adaptive",
                "--min-samples",
                "10",
                "--max-sweeps",
                "500",
                "--min-therm",
                "8",
                "--max-therm",
                "50",
                "--c-window",
                "4.0",
                "--tau-multiplier",
                "1.5",
            ],
            env=PLAIN_ENV,
        )
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "Adaptive" in plain
        assert "enabled" in plain
        # tau_int / interval columns of the adaptive results table
        assert "tau_int" in plain

    def test_run_correlation_adds_xi_column(self) -> None:
        result = runner.invoke(
            app, ["run", *FAST, "-T", "3.0", "--correlation"], env=PLAIN_ENV
        )
        assert result.exit_code == 0, result.output
        assert "xi" in _plain(result.output)


class TestRunCorrelationInterval:
    def test_flag_reaches_the_config_and_thins_the_series(self, tmp_path: Path) -> None:
        out = tmp_path / "corr.h5"
        result = runner.invoke(
            app,
            [
                "run",
                *FAST,
                "-T",
                "3.0",
                "--correlation",
                "--correlation-interval",
                "2",
                "-o",
                str(out),
            ],
            env=PLAIN_ENV,
        )
        assert result.exit_code == 0, result.output
        assert "Correlation interval" in _plain(result.output)
        with h5py.File(out, "r") as f:
            raw = f["metadata"].attrs["config_json"]
            config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            assert config["correlation_interval"] == 2
            # FAST: 8 sweeps at interval 4 = 2 measurements -> one evaluation.
            assert f["T=3.000000"]["correlation_length"].shape == (1,)

    def test_interval_beyond_the_measurements_is_rejected(self) -> None:
        result = runner.invoke(
            app,
            ["run", *FAST, "-T", "3.0", "--correlation", "--correlation-interval", "3"],
            env=PLAIN_ENV,
        )
        assert result.exit_code != 0
        assert "correlation_interval" in str(result.exception)
