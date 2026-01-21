"""Unit tests for MissingnessHeatmapPlot."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg", force=True)
import pandas as pd

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"
PLOTS_PATH = IAML_PATH / "plots"
STATS_PATH = IAML_PATH / "statistics"
MISSINGNESS_PATH = PLOTS_PATH / r"missingness_heatmap_plot.py"
MISSING_RATE_PATH = STATS_PATH / r"missing_rate_statistic.py"


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = [str(path)]


def _load_module(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_iaml_packages() -> None:
    _ensure_package("iaml", IAML_PATH)
    _ensure_package("iaml.plots", PLOTS_PATH)
    _ensure_package("iaml.statistics", STATS_PATH)


def _load_dataset_class():
    _prepare_iaml_packages()
    _load_module("iaml.data_type", IAML_PATH / "data_type.py")
    return _load_module("iaml.dataset", IAML_PATH / "dataset.py").Dataset


def _load_missingness_plot_class():
    _prepare_iaml_packages()
    _load_module("iaml.data_type", IAML_PATH / "data_type.py")
    _load_module("iaml.plot", IAML_PATH / "plot.py")
    _load_module("iaml.dataset", IAML_PATH / "dataset.py")
    module = _load_module("iaml.plots.missingness_heatmap_plot", MISSINGNESS_PATH)
    return module.MissingnessHeatmapPlot


def _load_missing_rate_statistic_class():
    _prepare_iaml_packages()
    _load_module("iaml.data_type", IAML_PATH / "data_type.py")
    _load_module("iaml.dataset", IAML_PATH / "dataset.py")
    _load_module("iaml.statistic", IAML_PATH / "statistic.py")
    module = _load_module("iaml.statistics.missing_rate_statistic", MISSING_RATE_PATH)
    return module.MissingRateStatistic


Dataset = _load_dataset_class()
MissingnessHeatmapPlot = _load_missingness_plot_class()
MissingRateStatistic = _load_missing_rate_statistic_class()


class TestMissingnessHeatmapPlot(unittest.TestCase):
    """Coverage for MissingnessHeatmapPlot core behavior."""

    def make_classification_dataset(
        self,
        n_samples: int = 120,
        seed: int = 20,
        n_classes: int = 2,
    ) -> Dataset:
        X, y = make_statistics_classification_data(
            n_samples=n_samples, seed=seed, n_classes=n_classes
        )
        return Dataset(X, y)

    def make_regression_dataset(
        self,
        n_samples: int = 120,
        seed: int = 30,
    ) -> Dataset:
        X, y = make_statistics_regression_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def make_survival_dataset(
        self,
        n_samples: int = 120,
        seed: int = 40,
    ) -> Dataset:
        X, y = make_statistics_survival_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def compute_statistic(self, dataset: Dataset) -> pd.DataFrame:
        statistic = MissingRateStatistic()
        return statistic.compute(dataset)

    def compute_plot(self, stats_df: pd.DataFrame, dataset: Dataset) -> MissingnessHeatmapPlot:
        plot = MissingnessHeatmapPlot()
        plot.compute(stats_df, dataset=dataset)
        return plot

    def assert_plot_image(self, plot: MissingnessHeatmapPlot) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def test_classification_missingness_heatmap(self) -> None:
        dataset = self.make_classification_dataset(n_samples=24, seed=12, n_classes=3)
        stats_df = self.compute_statistic(dataset)

        self.assertIn("missing_rate", stats_df.index)
        plot = self.compute_plot(stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_continuous_missingness_heatmap(self) -> None:
        dataset = self.make_regression_dataset(n_samples=20, seed=22)
        stats_df = self.compute_statistic(dataset)

        self.assertIn("missing_rate", stats_df.index)
        plot = self.compute_plot(stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_dataset_fallback_when_stats_missing(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=31)
        stats_df = pd.DataFrame({"num_1": [0.0]}, index=["mean"])

        plot = self.compute_plot(stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_survival_placeholder(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=17)
        plot = self.compute_plot(pd.DataFrame(), dataset=dataset)

        self.assert_plot_image(plot)
