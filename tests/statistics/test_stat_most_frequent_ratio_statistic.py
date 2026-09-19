"""Unit tests for MostFrequentRatioStatistic."""
from __future__ import annotations

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

_load_module("iaml.cache_keys", IAML_PATH / "cache_keys.py")
_load_module("iaml.data_type", IAML_PATH / "data_type.py")
_load_module("iaml.type_of_target", IAML_PATH / "type_of_target.py")
_load_module("iaml.logger", IAML_PATH / "logger.py")
_load_module("iaml.reference", IAML_PATH / "reference.py")
_load_module("iaml.dataset", IAML_PATH / "dataset.py")
_load_module("iaml.statistic", IAML_PATH / "statistic.py")

from tests.statistics.statistic_test_case import StatisticTestCase

DataType = sys.modules["iaml.data_type"].DataType

MostFrequentRatioStatistic = _load_module(
    "iaml.statistics.most_frequent_ratio",
    IAML_PATH / "statistics" / r"most_frequent_ratio.py",
).MostFrequentRatioStatistic


class TestMostFrequentRatioStatistic(StatisticTestCase):
    """Coverage for MostFrequentRatioStatistic core behavior."""

    def _categorical_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] == DataType.CATEGORICAL
        ]

    def _most_frequent_ratio(self, series: pd.Series) -> float:
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return 0.0
        total = int(counts.sum())
        if total <= 0:
            return 0.0
        top = int(counts.max())
        return top / total

    def test_classification_columns_and_values(self) -> None:
        dataset = self.make_classification_dataset(n_samples=36, seed=6, n_classes=3)
        result = self.compute_statistic(MostFrequentRatioStatistic, dataset)

        self.assertEqual(list(result.index), ["most_frequent_ratio"])

        categorical_cols = self._categorical_columns(dataset)
        self.assertTrue(categorical_cols)

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in categorical_cols
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        sample_col = categorical_cols[0]
        expected_all = self._most_frequent_ratio(dataset.X[sample_col])
        self.assertAlmostEqual(
            result.loc["most_frequent_ratio", f"{sample_col}_all"],
            expected_all,
        )
        for label in class_labels:
            values = dataset.X.loc[dataset.y == label, sample_col]
            expected = self._most_frequent_ratio(values)
            self.assertAlmostEqual(
                result.loc["most_frequent_ratio", f"{sample_col}_{label}"],
                expected,
            )

    def test_continuous_excludes_text_and_values(self) -> None:
        dataset = self.make_regression_dataset(n_samples=40, seed=11)
        result = self.compute_statistic(MostFrequentRatioStatistic, dataset)

        self.assertEqual(list(result.index), ["most_frequent_ratio"])

        categorical_cols = self._categorical_columns(dataset)
        self.assertEqual(set(result.columns), set(categorical_cols))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        sample_col = categorical_cols[0]
        expected = self._most_frequent_ratio(dataset.X[sample_col])
        self.assertAlmostEqual(
            result.loc["most_frequent_ratio", sample_col],
            expected,
        )

    def test_category_dtype_included(self) -> None:
        X = pd.DataFrame(
            {
                "cat_dtype": ["a", "a", "b", "c"],
                "num": [1, 2, 3, 4],
            }
        )
        y = [0.0, 1.0, 2.0, 3.0]
        dataset = self.make_dataset(X, y)
        dataset.X["cat_dtype"] = dataset.X["cat_dtype"].astype("category")

        result = self.compute_statistic(MostFrequentRatioStatistic, dataset)

        self.assertEqual(list(result.index), ["most_frequent_ratio"])
        self.assertIn("cat_dtype", result.columns)
        self.assertNotIn("num", result.columns)

        expected = self._most_frequent_ratio(dataset.X["cat_dtype"])
        self.assertAlmostEqual(
            result.loc["most_frequent_ratio", "cat_dtype"],
            expected,
        )

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=20, seed=13)
        result = self.compute_statistic(MostFrequentRatioStatistic, dataset)

        self.assertTrue(result.empty)
