"""Tests for plotting helpers driven by loaded HDF5 files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mcising.config import LatticeConfig, SimulationConfig
from mcising.io import load_hdf5, save_hdf5
from mcising.plotting import _export_prefix, plot_energy
from mcising.simulation import Simulation


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
