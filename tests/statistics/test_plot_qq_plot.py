"""Unit tests for QQPlot."""
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
QQ_PATH = PLOTS_PATH / r"qq_plot.py"


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
from iaml.data_type import DataType


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


def _load_qq_plot_class():
    _ensure_package("iaml.plots", PLOTS_PATH)
    return _load_class("iaml.plots.qq_plot", QQ_PATH, "QQPlot")


QQPlot = _load_qq_plot_class()


class TestQQPlot(StatisticPlotTestCase):
    """Coverage for QQPlot core behavior."""

    def _make_qq_stats(self, dataset) -> pd.DataFrame:
        numeric_columns = set(dataset.get_columns_names_by_type(DataType.NUMERIC))
        stats_row: dict[str, object] = {}
        for col in dataset.X.columns:
            if col in numeric_columns:
                stats_row[col] = {"values": dataset.X[col].to_numpy()}
            else:
                stats_row[col] = None
        return pd.DataFrame(stats_row, index=["qqplot"])

    def test_qq_plot_from_values(self) -> None:
        dataset = self.make_regression_dataset(n_samples=30, seed=11)
        stats_df = self._make_qq_stats(dataset)

        plot = self.compute_plot(QQPlot, stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_missing_qq_row_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1": [0.1]}, index=["mean"])

        plot = self.compute_plot(QQPlot, stats_df)

        self.assert_plot_image(plot)

    def test_no_valid_numeric_entries_placeholder(self) -> None:
        dataset = self.make_regression_dataset(n_samples=12, seed=13)
        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        stats_row = {
            col: {"theoretical": [0.0, 1.0], "ordered": [0.0]}
            for col in numeric_columns
        }
        stats_df = pd.DataFrame(stats_row, index=["qqplot"])

        plot = self.compute_plot(QQPlot, stats_df, dataset=dataset)

        self.assert_plot_image(plot)


if __name__ == "__main__":
    unittest.main()
