"""Unit tests for GroupedMeanStatistic."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAML_PATH = PROJECT_ROOT / "src" / "iaml"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.statistics", IAML_PATH / "statistics")

DataType = importlib.import_module("iaml.data_type").DataType

GroupedMeanStatistic = _load_module(
    "iaml.statistics.grouped_mean_statistic",
    IAML_PATH / "statistics" / r"grouped_mean_statistic.py",
).GroupedMeanStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


class TestGroupedMeanStatistic(StatisticTestCase):
    """Coverage for GroupedMeanStatistic core behavior."""

    def test_classification_grouped_mean_offsets_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=24, seed=25, n_classes=3)
        result = self.compute_statistic(GroupedMeanStatistic, dataset)

        self.assertEqual(list(result.index), ["grouped_mean"])

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in numeric_columns
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        overall = dataset.X["num_1"].mean()
        self.assertAlmostEqual(result.loc["grouped_mean", "num_1_all"], overall)

        sample_label = class_labels[0]
        class_mean = dataset.X.loc[dataset.y == sample_label, "num_1"].mean()
        self.assertAlmostEqual(
            result.loc["grouped_mean", f"num_1_{sample_label}"],
            class_mean - overall,
        )

        non_numeric = set(dataset.X.columns) - set(numeric_columns)
        for col in non_numeric:
            self.assertFalse(any(name.startswith(f"{col}_") for name in result.columns))

    def test_continuous_returns_empty(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=31)
        result = self.compute_statistic(GroupedMeanStatistic, dataset)

        self.assertTrue(result.empty)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=41)
        result = self.compute_statistic(GroupedMeanStatistic, dataset)

        self.assertTrue(result.empty)
