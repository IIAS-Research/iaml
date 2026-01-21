"""Unit tests for HistogramPlot."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.helpers.datasets import make_statistics_regression_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"
HISTOGRAM_PATH = IAML_PATH / "plots" / r"histogram_plot.py"


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if not hasattr(module, "__path__"):
        module.__path__ = [str(path)]
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
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


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.plots", IAML_PATH / "plots")

if "iaml.dataset" not in sys.modules:
    dataset_module = types.ModuleType("iaml.dataset")

    class Dataset:  # pragma: no cover - stub for type hints
        pass

    dataset_module.Dataset = Dataset
    sys.modules["iaml.dataset"] = dataset_module

_load_module("iaml.data_type", IAML_PATH / "data_type.py")
_load_module("iaml.plot", IAML_PATH / "plot.py")
_histogram_module = _load_module("iaml.plots.histogram_plot", HISTOGRAM_PATH)
HistogramPlot = _histogram_module.HistogramPlot


class TestHistogramPlot(unittest.TestCase):
    """Coverage for HistogramPlot core behavior."""

    def test_numeric_histogram_plot(self) -> None:
        X, _ = make_statistics_regression_data(n_samples=24, seed=31)
        stats_df = pd.DataFrame({col: [X[col].to_numpy()] for col in X.columns}, index=["histogram"])

        plot = HistogramPlot()
        plot.compute(stats_df)

        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def test_missing_histogram_row_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1": [1.0]}, index=["mean"])

        plot = HistogramPlot()
        plot.compute(stats_df)

        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def test_invalid_histogram_data_renders(self) -> None:
        stats_df = pd.DataFrame(
            {
                "num_1": [
                    {
                        "counts": np.array([1, 2, 3]),
                        "bins": np.array([0, 1, 2]),
                    }
                ]
            },
            index=["histogram"],
        )

        plot = HistogramPlot()
        plot.compute(stats_df)

        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)


if __name__ == "__main__":
    unittest.main()
