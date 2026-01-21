"""Unit tests for CorrelationHeatmapPlot."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg", force=True)
import pandas as pd

from tests.helpers.datasets import make_statistics_regression_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"
PLOTS_PATH = IAML_PATH / "plots"
CORRELATION_PATH = PLOTS_PATH / r"correlation_heatmap_plot.py"


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


def _load_correlation_plot_class():
    _ensure_package("iaml", IAML_PATH)
    _ensure_package("iaml.plots", PLOTS_PATH)
    _load_module("iaml.data_type", IAML_PATH / "data_type.py")
    _load_module("iaml.plot", IAML_PATH / "plot.py")
    _load_module("iaml.dataset", IAML_PATH / "dataset.py")
    module = _load_module("iaml.plots.correlation_heatmap_plot", CORRELATION_PATH)
    return module.CorrelationHeatmapPlot


def _load_dataset_class():
    _ensure_package("iaml", IAML_PATH)
    return _load_module("iaml.dataset", IAML_PATH / "dataset.py").Dataset


CorrelationHeatmapPlot = _load_correlation_plot_class()
Dataset = _load_dataset_class()


class TestCorrelationHeatmapPlot(unittest.TestCase):
    """Coverage for CorrelationHeatmapPlot core behavior."""

    def compute_plot(self, plot_cls, stats_df: pd.DataFrame, **kwargs):
        plot = plot_cls()
        plot.compute(stats_df, **kwargs)
        return plot

    def assert_plot_image(self, plot) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def make_regression_dataset(self, n_samples: int = 18, seed: int = 31):
        X, y = make_statistics_regression_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def test_square_dataframe_heatmap(self) -> None:
        stats_df = pd.DataFrame(
            [[1.0, 0.25], [0.25, 1.0]],
            index=["num_1", "num_2"],
            columns=["num_1", "num_2"],
        )

        plot = self.compute_plot(CorrelationHeatmapPlot, stats_df)

        self.assert_plot_image(plot)

    def test_correlation_row_dict_matrix(self) -> None:
        stats_df = pd.DataFrame(
            {
                "corr": [
                    {
                        "matrix": [[1.0, -0.3], [-0.3, 1.0]],
                        "labels": ["num_1", "num_2"],
                    }
                ]
            },
            index=["correlation"],
        )

        plot = self.compute_plot(CorrelationHeatmapPlot, stats_df)

        self.assert_plot_image(plot)

    def test_dataset_fallback_for_numeric_columns(self) -> None:
        dataset = self.make_regression_dataset(n_samples=18, seed=31)
        stats_df = pd.DataFrame({"num_1": [1.0]}, index=["mean"])

        plot = self.compute_plot(CorrelationHeatmapPlot, stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_empty_dataframe_placeholder(self) -> None:
        plot = self.compute_plot(CorrelationHeatmapPlot, pd.DataFrame())

        self.assert_plot_image(plot)
