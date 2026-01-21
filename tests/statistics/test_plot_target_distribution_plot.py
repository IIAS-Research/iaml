"""Unit tests for TargetDistributionPlot."""
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


def _resolve_target_plot_path() -> Path:
    explicit = PLOTS_PATH / "target_distribution_plot.py"
    if explicit.exists():
        return explicit
    candidates = sorted(PLOTS_PATH.glob("targe*lot.py"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Target distribution plot module not found in {PLOTS_PATH}")


TARGET_PATH = _resolve_target_plot_path()


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


def _load_dataset_class():
    _ensure_package("iaml", IAML_PATH)
    _load_module("iaml.cache_keys", IAML_PATH / "cache_keys.py")
    _load_module("iaml.data_type", IAML_PATH / "data_type.py")
    _load_module("iaml.type_of_target", IAML_PATH / "type_of_target.py")
    _load_module("iaml.logger", IAML_PATH / "logger.py")
    module = _load_module("iaml.dataset", IAML_PATH / "dataset.py")
    return module.Dataset


def _load_target_distribution_plot_class():
    _ensure_package("iaml", IAML_PATH)
    _ensure_package("iaml.plots", PLOTS_PATH)
    _load_module("iaml.plot", IAML_PATH / "plot.py")
    _load_module("iaml.dataset", IAML_PATH / "dataset.py")
    module = _load_module("iaml.plots.target_distribution_plot", TARGET_PATH)
    return module.TargetDistributionPlot


Dataset = _load_dataset_class()
TargetDistributionPlot = _load_target_distribution_plot_class()


class TestTargetDistributionPlot(unittest.TestCase):
    """Coverage for TargetDistributionPlot core behavior."""

    def _make_dataset(self, factory, **kwargs):
        X, y = factory(**kwargs)
        return Dataset(X, y)

    def _compute_plot(self, stats_df: pd.DataFrame, dataset) -> TargetDistributionPlot:
        plot = TargetDistributionPlot()
        plot.compute(stats_df, dataset=dataset)
        return plot

    def _assert_plot_image(self, plot: TargetDistributionPlot) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def test_value_counts_from_stats(self) -> None:
        dataset = self._make_dataset(
            make_statistics_classification_data,
            n_samples=18,
            seed=12,
            n_classes=3,
        )
        counts = pd.Series(dataset.y).value_counts(dropna=False).to_dict()
        stats_df = pd.DataFrame({"target": [counts]}, index=["value_counts"])

        plot = self._compute_plot(stats_df, dataset=dataset)

        self._assert_plot_image(plot)

    def test_histogram_from_stats(self) -> None:
        dataset = self._make_dataset(make_statistics_regression_data, n_samples=22, seed=27)
        stats_df = pd.DataFrame({"target": [dataset.y]}, index=["histogram"])

        plot = self._compute_plot(stats_df, dataset=dataset)

        self._assert_plot_image(plot)

    def test_survival_placeholder(self) -> None:
        dataset = self._make_dataset(make_statistics_survival_data, n_samples=16, seed=41)

        plot = self._compute_plot(pd.DataFrame(), dataset=dataset)

        self._assert_plot_image(plot)


if __name__ == "__main__":
    unittest.main()
