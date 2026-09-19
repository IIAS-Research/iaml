"""Unit tests for ViolinPlot."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg", force=True)
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = [str(path)]


def _load_class(module_name: str, module_path: Path, class_name: str):
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {class_name} module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def _load_violin_plot_class():
    plots_path = SRC_PATH / "iaml" / "plots"
    violin_path = plots_path / r"violin_plot.py"

    _ensure_package("iaml", SRC_PATH / "iaml")
    _ensure_package("iaml.plots", plots_path)

    return _load_class("iaml.plots.violin_plot", violin_path, "ViolinPlot")


ViolinPlot = _load_violin_plot_class()


def _make_violin_stats() -> dict:
    return {
        "A": {
            "density": [0.1, 0.2, 0.1],
            "support": [0.0, 1.0, 2.0],
            "quartiles": [0.5, 1.0, 1.5],
        },
        "B": {
            "density": [0.2, 0.1, 0.2],
            "support": [0.0, 1.0, 2.0],
            "quartiles": [0.4, 1.1, 1.6],
        },
    }


class TestViolinPlot(unittest.TestCase):
    """Coverage for ViolinPlot core behavior."""

    def compute_plot(self, stats_df: pd.DataFrame) -> ViolinPlot:
        plot = ViolinPlot()
        plot.compute(stats_df)
        return plot

    def assert_plot_image(self, plot: ViolinPlot) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def test_continuous_violin_plot(self) -> None:
        stats_df = pd.DataFrame({"cat_feature": [_make_violin_stats()]}, index=["violin"])
        plot = self.compute_plot(stats_df)

        self.assert_plot_image(plot)

    def test_missing_violin_row_placeholder(self) -> None:
        stats_df = pd.DataFrame({"cat_small": ["nope"]}, index=["mean"])
        plot = self.compute_plot(stats_df)

        self.assert_plot_image(plot)

    def test_no_violin_groups_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1": [None], "num_2": [None]}, index=["violin"])
        plot = self.compute_plot(stats_df)

        self.assert_plot_image(plot)
