"""Unit tests for RareCategoryRateStatistic."""
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

RareCategoryRateStatistic = _load_module(
    "iaml.statistics.rare_category_rate",
    IAML_PATH / "statistics" / r"rare_category_rate.py",
).RareCategoryRateStatistic


class TestRareCategoryRateStatistic(StatisticTestCase):
    """Coverage for RareCategoryRateStatistic core behavior."""

    def _categorical_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] == DataType.CATEGORICAL
        ]

    def _rare_ratio(self, series: pd.Series, threshold: float) -> float:
        counts = series.value_counts(dropna=True)
        total = int(counts.sum())
        if total <= 0 or counts.empty:
            return 0.0
        if threshold <= 0:
            return 0.0
        ratios = counts / total
        rare_count = int((ratios < threshold).sum())
        return rare_count / len(counts)

    def test_classification_columns_and_values(self) -> None:
        dataset = self.make_classification_dataset(n_samples=36, seed=6, n_classes=3)
        result = self.compute_statistic(RareCategoryRateStatistic, dataset)

        self.assertEqual(list(result.index), ["rare_category_rate"])

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
        threshold = 0.01
        expected_all = self._rare_ratio(dataset.X[sample_col], threshold)
        self.assertAlmostEqual(
            result.loc["rare_category_rate", f"{sample_col}_all"],
            expected_all,
        )
        for label in class_labels:
            values = dataset.X.loc[dataset.y == label, sample_col]
            expected = self._rare_ratio(values, threshold)
            self.assertAlmostEqual(
                result.loc["rare_category_rate", f"{sample_col}_{label}"],
                expected,
            )

    def test_continuous_excludes_text_and_values(self) -> None:
        dataset = self.make_regression_dataset(n_samples=40, seed=11)
        result = self.compute_statistic(RareCategoryRateStatistic, dataset)

        self.assertEqual(list(result.index), ["rare_category_rate"])

        categorical_cols = self._categorical_columns(dataset)
        self.assertEqual(set(result.columns), set(categorical_cols))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        sample_col = categorical_cols[0]
        expected = self._rare_ratio(dataset.X[sample_col], 0.01)
        self.assertAlmostEqual(
            result.loc["rare_category_rate", sample_col],
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

        statistic = RareCategoryRateStatistic()
        result = statistic.compute(dataset, threshold=0.4)

        self.assertEqual(list(result.index), ["rare_category_rate"])
        self.assertIn("cat_dtype", result.columns)
        self.assertNotIn("num", result.columns)

        expected = self._rare_ratio(dataset.X["cat_dtype"], 0.4)
        self.assertAlmostEqual(
            result.loc["rare_category_rate", "cat_dtype"],
            expected,
        )

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=20, seed=13)
        result = self.compute_statistic(RareCategoryRateStatistic, dataset)

        self.assertTrue(result.empty)
