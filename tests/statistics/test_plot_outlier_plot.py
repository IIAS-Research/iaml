"""Unit tests for OutlierPlot."""
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
STATS_PATH = SRC_PATH / "iaml" / "statistics"
OUTLIER_PLOT_PATH = PLOTS_PATH / r"outlier_plot.py"
OUTLIER_STAT_PATH = STATS_PATH / "outlier_count_iqr_statistic.py"


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


def _load_outlier_plot_class():
    _ensure_package("iaml.plots", PLOTS_PATH)
    return _load_class("iaml.plots.outlier_plot", OUTLIER_PLOT_PATH, "OutlierPlot")


def _load_outlier_stat_class():
    _ensure_package("iaml.statistics", STATS_PATH)
    return _load_class(
        "iaml.statistics.outlier_count_iqr_statistic",
        OUTLIER_STAT_PATH,
        "OutlierCountIQRStatistic",
    )


OutlierPlot = _load_outlier_plot_class()
OutlierCountIQRStatistic = _load_outlier_stat_class()


class TestOutlierPlot(StatisticPlotTestCase):
    """Coverage for OutlierPlot core behavior."""

    def _make_outlier_stats(self, dataset) -> pd.DataFrame:
        return self.compute_statistic(OutlierCountIQRStatistic, dataset)

    def test_classification_outlier_plot(self) -> None:
        dataset = self.make_classification_dataset(n_samples=30, seed=23, n_classes=3)
        stats_df = self._make_outlier_stats(dataset)

        plot = self.compute_plot(OutlierPlot, stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_continuous_outlier_plot(self) -> None:
        dataset = self.make_regression_dataset(n_samples=25, seed=17)
        stats_df = self._make_outlier_stats(dataset)

        plot = self.compute_plot(OutlierPlot, stats_df, dataset=dataset, base_name="Demo")

        self.assert_plot_image(plot)

    def test_missing_outlier_row_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1": [1.0]}, index=["mean"])

        plot = self.compute_plot(OutlierPlot, stats_df)

        self.assert_plot_image(plot)


if __name__ == "__main__":
    unittest.main()
