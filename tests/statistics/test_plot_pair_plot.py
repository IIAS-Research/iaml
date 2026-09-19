"""Unit tests for PairPlot."""
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
PLOTS_PATH = SRC_PATH / "iaml" / "plots"
PAIR_PLOT_PATH = PLOTS_PATH / r"pair_plot.py"


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = [str(path)]


# Avoid importing iaml/__init__.py, which currently raises a SyntaxError.
_ensure_package("iaml", SRC_PATH / "iaml")

from tests.statistics.plot_test_case import StatisticPlotTestCase


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


def _load_pair_plot_class():
    _ensure_package("iaml.plots", PLOTS_PATH)
    return _load_class("iaml.plots.pair_plot", PAIR_PLOT_PATH, "PairPlot")


PairPlot = _load_pair_plot_class()


class TestPairPlot(StatisticPlotTestCase):
    """Coverage for PairPlot core behavior."""

    def _make_pair_stats(self, dataset) -> pd.DataFrame:
        numeric_columns = dataset.X.select_dtypes(include="number").columns
        stats_row: dict[str, object] = {}
        for col in dataset.X.columns:
            if col in numeric_columns:
                stats_row[col] = {"values": dataset.X[col].to_numpy()}
            else:
                stats_row[col] = None
        return pd.DataFrame(stats_row, index=["pairplot"])

    def test_pair_plot_from_stats_row(self) -> None:
        dataset = self.make_regression_dataset(n_samples=24, seed=11)
        stats_df = self._make_pair_stats(dataset)

        plot = self.compute_plot(PairPlot, stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_pair_plot_falls_back_to_dataset(self) -> None:
        dataset = self.make_regression_dataset(n_samples=20, seed=21)

        plot = self.compute_plot(PairPlot, pd.DataFrame(), dataset=dataset)

        self.assert_plot_image(plot)

    def test_pair_plot_placeholder_without_numeric_data(self) -> None:
        stats_df = pd.DataFrame({"cat": ["A"], "num_1": [1.0]}, index=["mean"])

        plot = self.compute_plot(PairPlot, stats_df)

        self.assert_plot_image(plot)


if __name__ == "__main__":
    unittest.main()
