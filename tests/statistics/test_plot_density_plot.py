"""Unit tests for DensityPlot."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Type

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd

from tests.helpers.datasets import make_statistics_regression_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
PLOTS_PATH = SRC_PATH / "iaml" / "plots"
DENSITY_PATH = PLOTS_PATH / r"density_plot.py"
DATASET_PATH = SRC_PATH / "iaml" / "dataset.py"


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


def _load_density_plot_class():
    _ensure_package("iaml", SRC_PATH / "iaml")
    _ensure_package("iaml.plots", PLOTS_PATH)
    return _load_class("iaml.plots.density_plot", DENSITY_PATH, "DensityPlot")


DensityPlot = _load_density_plot_class()
Dataset = _load_class("iaml.dataset", DATASET_PATH, "Dataset")


class TestDensityPlot(unittest.TestCase):
    """Coverage for DensityPlot core behavior."""

    def make_dataset(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        return Dataset(X, y)

    def compute_plot(self, plot_cls: Type[Any], stats_df: pd.DataFrame, **kwargs) -> Any:
        plot = plot_cls()
        plot.compute(stats_df, **kwargs)
        return plot

    def assert_plot_image(self, plot: Any) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def test_density_plot_from_values(self) -> None:
        X, y = make_statistics_regression_data(n_samples=40, seed=11)
        dataset = self.make_dataset(X, y)
        stats_row: dict[str, object] = {}
        for col in X.columns:
            if col.startswith("num_"):
                stats_row[col] = {"values": X[col].to_numpy()}
            else:
                stats_row[col] = None
        stats_df = pd.DataFrame(stats_row, index=["density"])

        plot = self.compute_plot(DensityPlot, stats_df, dataset=dataset)

        self.assert_plot_image(plot)

    def test_density_plot_with_precomputed_support(self) -> None:
        X, y = make_statistics_regression_data(n_samples=24, seed=17)
        dataset = self.make_dataset(X, y)
        values = X["num_1"].dropna().to_numpy()
        counts, edges = np.histogram(values, bins=6, density=True)
        support = (edges[:-1] + edges[1:]) / 2.0
        stats_df = pd.DataFrame(
            {"num_1": [{"support": support, "density": counts}]},
            index=["density"],
        )

        plot = self.compute_plot(DensityPlot, stats_df, dataset=dataset, base_name="num_1")

        self.assert_plot_image(plot)

    def test_missing_density_row_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1": [1.0]}, index=["mean"])

        plot = self.compute_plot(DensityPlot, stats_df)

        self.assert_plot_image(plot)


if __name__ == "__main__":
    unittest.main()
