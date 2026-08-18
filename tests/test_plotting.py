"""Tests for plotting helpers driven by loaded HDF5 files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from mcising.config import LatticeConfig, SimulationConfig
from mcising.io import load_hdf5, save_hdf5
from mcising.plotting import (
    _export_prefix,
    plot_energy,
    plot_specific_heat,
    plot_susceptibility,
)
from mcising.simulation import Simulation, SimulationResults


def _run_and_save(tmp_path: Path, name: str, j1: float) -> Path:
    config = SimulationConfig(
        lattice=LatticeConfig(size=4, j1=j1),
        temperatures=(3.0, 2.0),
        n_sweeps=20,
        measurement_interval=10,
    )
    results = Simulation(config).run(show_progress=False)
    path = tmp_path / name
    save_hdf5(results, path)
    return path


class TestLoadedFileLegends:
    """Gate: plots from loaded files carry real legend labels (B12).

    Before P07, load_hdf5 never restored the config, so _label_for
    returned "" and multi-file overlays rendered an empty legend.
    """

    def test_legend_labels_non_empty_from_loaded_files(self, tmp_path: Path) -> None:
        p1 = _run_and_save(tmp_path, "fm.h5", 1.0)
        p2 = _run_and_save(tmp_path, "afm.h5", -1.0)

        fig = plot_energy([p1, p2])
        try:
            legend = fig.axes[0].get_legend()
            assert legend is not None
            texts = [t.get_text() for t in legend.get_texts()]
            assert len(texts) == 2
            assert all(texts), f"empty legend label in {texts!r}"
            assert any("J1=1.0" in t for t in texts)
            assert any("J1=-1.0" in t for t in texts)
        finally:
            plt.close(fig)


class TestExportPrefix:
    def test_export_prefix_from_loaded_file(self, tmp_path: Path) -> None:
        # The generic "mcising" prefix was the loaded-file symptom of the
        # missing config; a restored config rebuilds the descriptive one.
        path = _run_and_save(tmp_path, "run.h5", 1.0)
        loaded = load_hdf5(path)
        prefix = _export_prefix(loaded)
        assert prefix != "mcising"
        assert prefix == "square_4x4_J1=1.0_metropolis"


def _run_results(n_sweeps: int = 400) -> SimulationResults:
    config = SimulationConfig(
        lattice=LatticeConfig(size=8),
        temperatures=(3.0, 2.0),
        n_sweeps=n_sweeps,
        measurement_interval=10,
    )
    return Simulation(config).run(show_progress=False)


def _errorbar_half_heights(fig: Figure) -> list[float]:
    """Vertical half-extents of the error bars in the first axes."""
    containers = [
        c for c in fig.axes[0].containers if isinstance(c, ErrorbarContainer)
    ]
    assert containers, "no ErrorbarContainer on the axes"
    assert containers[0].has_yerr
    barlinecols = containers[0][2]
    segments = barlinecols[0].get_segments()
    return [
        abs(seg[1, 1] - seg[0, 1]) / 2.0
        for seg in segments
        if seg.shape == (2, 2)
    ]


class TestErrorBars:
    """Gate: real error bars in observable plots (B10).

    Before P08, Cv and chi plots hardcoded 0.0 errors and silently fell
    back to a bare line plot; E and M bars showed the sample spread,
    not a standard error.
    """

    def test_specific_heat_plot_has_nonzero_error_bars(self) -> None:
        results = _run_results()
        fig = plot_specific_heat(results)
        try:
            heights = _errorbar_half_heights(fig)
            assert heights, "no finite error-bar segments drawn"
            assert max(heights) > 0.0
        finally:
            plt.close(fig)

    def test_susceptibility_plot_has_nonzero_error_bars(self) -> None:
        results = _run_results()
        fig = plot_susceptibility(results)
        try:
            assert max(_errorbar_half_heights(fig)) > 0.0
        finally:
            plt.close(fig)

    def test_energy_bars_are_standard_errors_not_spread(self) -> None:
        results = _run_results()
        stats = results.statistics(2.0)
        fig = plot_energy(results)
        try:
            heights = _errorbar_half_heights(fig)
            # Temperatures plot in ascending order: index 0 is T=2.0.
            assert heights[0] == pytest.approx(stats.energy.error, rel=1e-9)
            # The old bars used np.std(series): the spread, larger than
            # the SE of the mean by ~sqrt(n_eff).
            spread = float(np.std(results.energy[2.0]))
            assert heights[0] < spread
        finally:
            plt.close(fig)

    def test_missing_temperature_does_not_break_plot(self) -> None:
        # Regression: temps came from results.temperatures while vals
        # skipped missing entries, so a partial results object produced
        # length-mismatched arrays.
        results = _run_results(n_sweeps=100)
        del results.energy[3.0]
        fig = plot_energy(results)
        try:
            line = fig.axes[0].lines[0]
            assert len(line.get_xdata()) == 1
        finally:
            plt.close(fig)

    def test_short_series_nan_error_still_plots(self, tmp_path: Path) -> None:
        # n=2 measurements: jackknife error is NaN by policy; the plot
        # must still render markers (no exception, no zero-height lie).
        path = _run_and_save(tmp_path, "short.h5", 1.0)
        fig = plot_specific_heat(load_hdf5(path))
        try:
            containers = [
                c
                for c in fig.axes[0].containers
                if isinstance(c, ErrorbarContainer)
            ]
            assert containers
            line = containers[0][0]
            assert len(line.get_xdata()) == 2
        finally:
            plt.close(fig)
