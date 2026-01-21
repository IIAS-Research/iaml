"""Unit tests for TopKValueCountsStatistic."""
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

TopKValueCountsStatistic = _load_module(
    "iaml.statistics.top_k_value_counts",
    IAML_PATH / "statistics" / r"top_k_value_counts.py",
).TopKValueCountsStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


class TestTopKValueCountsStatistic(StatisticTestCase):
    """Coverage for TopKValueCountsStatistic core behavior."""

    def _categorical_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] == DataType.CATEGORICAL
        ]

    def _top_k_counts(self, series: pd.Series, k: int) -> list[tuple[object, int, float]]:
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return []
        top_counts = counts.head(k)
        total = counts.sum()
        return [(idx, int(val), float(val) / total) for idx, val in top_counts.items()]

    def test_classification_top_k_counts_per_class(self) -> None:
        dataset = self.make_classification_dataset(n_samples=36, seed=6, n_classes=3)
        statistic = TopKValueCountsStatistic(k=2)
        result = statistic.compute(dataset)

        self.assertEqual(list(result.index), [str(statistic)])

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._categorical_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        for col in self._categorical_columns(dataset):
            expected_all = self._top_k_counts(dataset.X[col], statistic.k)
            self.assertEqual(result.loc[str(statistic), f"{col}_all"], expected_all)
            for label in class_labels:
                label_values = dataset.X.loc[dataset.y == label, col]
                expected = self._top_k_counts(label_values, statistic.k)
                self.assertEqual(
                    result.loc[str(statistic), f"{col}_{label}"],
                    expected,
                )

    def test_continuous_top_k_counts_categorical_only(self) -> None:
        dataset = self.make_regression_dataset(n_samples=40, seed=12)
        statistic = TopKValueCountsStatistic(k=3)
        result = statistic.compute(dataset)

        self.assertEqual(list(result.index), [str(statistic)])
        self.assertEqual(set(result.columns), set(self._categorical_columns(dataset)))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        for col in self._categorical_columns(dataset):
            expected = self._top_k_counts(dataset.X[col], statistic.k)
            self.assertEqual(result.loc[str(statistic), col], expected)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=24, seed=18)
        statistic = TopKValueCountsStatistic(k=2)
        result = statistic.compute(dataset)

        self.assertTrue(result.empty)
