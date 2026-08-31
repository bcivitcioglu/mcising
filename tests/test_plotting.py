"""Tests for plotting helpers driven by loaded HDF5 files."""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from mcising.config import LatticeConfig, LatticeType, SimulationConfig
from mcising.io import load_hdf5, save_hdf5
from mcising.plotting import (
    _export_prefix,
    export_lattices,
    plot_correlation,
    plot_energy,
    plot_energy_timeseries,
    plot_lattice,
    plot_magnetization_histogram,
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


def _quick_results(
    j1: float = 1.0, *, compute_correlation: bool = False, n_sweeps: int = 60
) -> SimulationResults:
    """A tiny 4x4 run: 6 configs per temperature at the default sweeps."""
    config = SimulationConfig(
        lattice=LatticeConfig(size=4, j1=j1),
        temperatures=(3.0, 2.0),
        n_sweeps=n_sweeps,
        measurement_interval=10,
        compute_correlation=compute_correlation,
    )
    return Simulation(config).run(show_progress=False)


class TestPlotQuantityContent:
    """Figure content of the vs-temperature plots, beyond error bars."""

    def test_energy_plot_axes_content(self) -> None:
        results = _quick_results()
        fig = plot_energy(results)
        try:
            ax = fig.axes[0]
            containers = [
                c for c in ax.containers if isinstance(c, ErrorbarContainer)
            ]
            assert len(containers) == 1
            assert ax.get_xlabel() == "Temperature"
            assert ax.get_ylabel() == "<E>/N"
            assert ax.get_title() == "Energy per site"
            line = containers[0][0]
            assert np.array_equal(line.get_xdata(), [2.0, 3.0])
        finally:
            plt.close(fig)

    def test_multi_results_container_count_and_legend(self) -> None:
        fm = _quick_results(1.0)
        afm = _quick_results(-1.0)
        fig = plot_energy([fm, afm])
        try:
            ax = fig.axes[0]
            containers = [
                c for c in ax.containers if isinstance(c, ErrorbarContainer)
            ]
            assert len(containers) == 2
            legend = ax.get_legend()
            assert legend is not None
            assert len(legend.get_texts()) == 2
        finally:
            plt.close(fig)

    def test_ax_reuse_draws_on_given_axes(self) -> None:
        # The ax= arm must draw on the caller's axes and hand back the
        # caller's figure, never a fresh one.
        results = _quick_results(compute_correlation=True)
        fig, axes = plt.subplots(2, 2)
        try:
            assert plot_energy(results, ax=axes[0][0]) is fig
            assert plot_energy_timeseries(results, 2.0, ax=axes[0][1]) is fig
            assert plot_magnetization_histogram(results, 2.0, ax=axes[1][0]) is fig
            assert plot_correlation(results, 2.0, ax=axes[1][1]) is fig
            assert axes[0][0].containers, "errorbar not drawn on the given axes"
            assert axes[0][1].lines, "timeseries not drawn on the given axes"
            assert axes[1][0].patches, "histogram not drawn on the given axes"
            assert axes[1][1].lines, "correlation not drawn on the given axes"
        finally:
            plt.close(fig)


class TestPlotCorrelation:
    def test_plot_correlation_content(self) -> None:
        results = _quick_results(compute_correlation=True)
        fig = plot_correlation(results, 2.0)
        try:
            ax = fig.axes[0]
            assert results.correlation_function is not None
            distances, correlations = results.correlation_function[2.0]
            line = ax.lines[0]
            assert np.array_equal(line.get_xdata(), distances)
            assert np.array_equal(line.get_ydata(), correlations)
            assert ax.get_xlabel() == "Distance (lattice units)"
            assert ax.get_ylabel() == "C(r)"
            assert ax.get_title() == "Correlation Function at T=2.000"
            # Second line is the y=0 guide.
            assert np.all(np.asarray(ax.lines[1].get_ydata()) == 0.0)
        finally:
            plt.close(fig)

    def test_plot_correlation_missing_temperature_raises(self) -> None:
        results = _quick_results(compute_correlation=True)
        with pytest.raises(ValueError, match="No correlation data"):
            plot_correlation(results, 5.0)

    def test_plot_correlation_without_correlation_raises(self) -> None:
        results = _quick_results()  # compute_correlation=False
        with pytest.raises(ValueError, match="No correlation data"):
            plot_correlation(results, 2.0)


class TestPlotLatticeRawArrays:
    """_render_spins dimensionality handling through the raw-array path."""

    def _image(self, fig: Figure) -> object:
        images = fig.axes[0].images
        assert images, "no image rendered"
        return images[0]

    def test_raw_1d_array_renders_as_row(self) -> None:
        rng = np.random.default_rng(0)
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=16)
        fig = plot_lattice(spins)
        try:
            image = fig.axes[0].images[0]
            assert image.get_array().shape == (1, 16)
        finally:
            plt.close(fig)

    def test_raw_3d_array_flattens_trailing_dims(self) -> None:
        rng = np.random.default_rng(0)
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, 4, 4))
        fig = plot_lattice(spins)
        try:
            image = fig.axes[0].images[0]
            assert image.get_array().shape == (4, 16)
        finally:
            plt.close(fig)

    def test_raw_2d_array_style(self) -> None:
        rng = np.random.default_rng(0)
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, 4))
        fig = plot_lattice(spins)
        try:
            ax = fig.axes[0]
            image = ax.images[0]
            assert image.get_array().shape == (4, 4)
            assert image.get_cmap().name == "RdBu"
            assert image.get_clim() == (-1.0, 1.0)
            assert len(ax.get_xticks()) == 0
            assert len(ax.get_yticks()) == 0
        finally:
            plt.close(fig)


class TestPlotLatticeGrid:
    def test_grid_all_configs_panel_count_and_titles(self) -> None:
        results = _quick_results()  # 6 configs per temperature
        fig = plot_lattice(results, temperature=2.0)
        try:
            visible = [ax for ax in fig.axes if ax.get_visible()]
            assert len(visible) == 6
            assert [ax.get_title() for ax in visible] == [str(i) for i in range(6)]
            assert fig.get_suptitle() == "T=2.0000  (6 configurations)"
        finally:
            plt.close(fig)

    def test_grid_max_panels_subsampling_and_blank_panels(self) -> None:
        # 6 configs into 4 panels: np.linspace(0, 5, 4) -> indices
        # [0, 1, 3, 5]; the 2x3 grid hides the two unused axes.
        results = _quick_results()
        fig = plot_lattice(results, temperature=2.0, max_panels=4, max_cols=3)
        try:
            assert len(fig.axes) == 6
            visible = [ax for ax in fig.axes if ax.get_visible()]
            assert len(visible) == 4
            assert [ax.get_title() for ax in visible] == ["0", "1", "3", "5"]
        finally:
            plt.close(fig)

    def test_single_config_mode(self) -> None:
        results = _quick_results()
        fig = plot_lattice(results, temperature=2.0, n=1)
        try:
            assert len(fig.axes) == 1
            assert fig.axes[0].get_title() == "T=2.0000, config 1"
        finally:
            plt.close(fig)

    def test_n_out_of_range_raises(self) -> None:
        results = _quick_results()
        with pytest.raises(ValueError, match="out of range"):
            plot_lattice(results, temperature=2.0, n=99)

    def test_default_temperature_is_median(self) -> None:
        # Two temperatures sorted [2.0, 3.0]: index len//2 = 1 -> T=3.0.
        results = _quick_results()
        fig = plot_lattice(results)
        try:
            assert fig.get_suptitle().startswith("T=3.0000")
        finally:
            plt.close(fig)

    def test_missing_configs_temperature_raises(self) -> None:
        results = _quick_results()
        with pytest.raises(ValueError, match="No configurations stored"):
            plot_lattice(results, temperature=5.0)


class TestExportLatticesSchema:
    """The zip/PNG schema contract of export_lattices."""

    def test_export_tree_schema(self, tmp_path: Path) -> None:
        results = _quick_results()
        out = tmp_path / "out.zip"
        count = export_lattices(results, out)
        prefix = _export_prefix(results)
        expected = {
            f"{prefix}/T={t:.4f}/config_{i:03d}.png"
            for t in (2.0, 3.0)
            for i in range(1, 7)  # config numbering is 1-based
        }
        with zipfile.ZipFile(out) as zf:
            assert set(zf.namelist()) == expected
        assert count == len(expected)

    def test_export_flat_schema(self, tmp_path: Path) -> None:
        results = _quick_results()
        out = tmp_path / "flat.zip"
        count = export_lattices(results, out, flat=True)
        prefix = _export_prefix(results)
        expected = {
            f"{prefix}/{prefix}_T={t:.4f}_config_{i:03d}.png"
            for t in (2.0, 3.0)
            for i in range(1, 7)
        }
        with zipfile.ZipFile(out) as zf:
            assert set(zf.namelist()) == expected
        assert count == len(expected)

    def test_export_temperatures_filter(self, tmp_path: Path) -> None:
        results = _quick_results()
        out = tmp_path / "filtered.zip"
        count = export_lattices(results, out, temperatures=[2.0])
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert count == len(names) == 6
        assert all("/T=2.0000/" in name for name in names)

    def test_export_members_are_real_pngs(self, tmp_path: Path) -> None:
        results = _quick_results()
        out = tmp_path / "png.zip"
        export_lattices(results, out, temperatures=[2.0])
        with zipfile.ZipFile(out) as zf:
            for name in zf.namelist():
                with zf.open(name) as member:
                    assert member.read(8) == b"\x89PNG\r\n\x1a\n", name


class TestExportPrefixBranches:
    """_export_prefix only reads config metadata — no runs needed."""

    @staticmethod
    def _prefix_for(lattice: LatticeConfig) -> str:
        config = SimulationConfig(
            lattice=lattice,
            temperatures=(2.0,),
            n_sweeps=10,
            measurement_interval=10,
        )
        return _export_prefix(SimulationResults(metadata={"config": config}))

    def test_prefix_cubic(self) -> None:
        lattice = LatticeConfig(lattice_type=LatticeType.CUBIC, size=4)
        assert self._prefix_for(lattice) == "cubic_4x4x4_J1=1.0_metropolis"

    def test_prefix_chain(self) -> None:
        lattice = LatticeConfig(lattice_type=LatticeType.CHAIN, size=8)
        assert self._prefix_for(lattice) == "chain_8_J1=1.0_metropolis"

    def test_prefix_multi_coupling(self) -> None:
        lattice = LatticeConfig(size=4, j1=0.0, j2=0.5, h=0.25)
        assert self._prefix_for(lattice) == "square_4x4_J2=0.5_h=0.25_metropolis"

    def test_prefix_all_zero_couplings(self) -> None:
        lattice = LatticeConfig(size=4, j1=0.0)
        assert self._prefix_for(lattice) == "square_4x4_J1=0_metropolis"

    def test_prefix_no_config_is_mcising(self) -> None:
        assert _export_prefix(SimulationResults()) == "mcising"
