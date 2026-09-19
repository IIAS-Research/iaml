"""Unit tests for CategoryCooccurrenceStatistic."""
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


def _cooccurrence(frame: pd.DataFrame, col_a: str, col_b: str) -> list[tuple[tuple, int]]:
    values = frame[[col_a, col_b]].dropna()
    if values.empty:
        return []
    counts = values.groupby([col_a, col_b], sort=True).size()
    return [((idx_a, idx_b), int(count)) for (idx_a, idx_b), count in counts.items()]


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.statistics", IAML_PATH / "statistics")

DataType = importlib.import_module("iaml.data_type").DataType

CategoryCooccurrenceStatistic = _load_module(
    "iaml.statistics.category_cooccurrence_statistic",
    IAML_PATH / "statistics" / r"category_cooccurrence_statistic.py",
).CategoryCooccurrenceStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


class TestCategoryCooccurrenceStatistic(StatisticTestCase):
    """Coverage for CategoryCooccurrenceStatistic core behavior."""

    def test_classification_cooccurrence_counts(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=12, n_classes=3)
        result = self.compute_statistic(CategoryCooccurrenceStatistic, dataset)

        self.assertEqual(list(result.index), ["category_cooccurrence"])

        categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        self.assertIn("cat_small", categorical_columns)
        self.assertIn("cat_with_nan", categorical_columns)

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"cat_small__cat_with_nan_{label}" for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        expected_all = _cooccurrence(dataset.X, "cat_small", "cat_with_nan")
        self.assertEqual(
            result.loc["category_cooccurrence", "cat_small__cat_with_nan_all"],
            expected_all,
        )

        sample_label = class_labels[0]
        expected_label = _cooccurrence(
            dataset.X.loc[dataset.y == sample_label],
            "cat_small",
            "cat_with_nan",
        )
        self.assertEqual(
            result.loc["category_cooccurrence", f"cat_small__cat_with_nan_{sample_label}"],
            expected_label,
        )

    def test_continuous_cooccurrence_counts(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=21)
        result = self.compute_statistic(CategoryCooccurrenceStatistic, dataset)

        self.assertEqual(list(result.index), ["category_cooccurrence"])
        self.assertEqual(set(result.columns), {"cat_small__cat_with_nan"})

        expected_all = _cooccurrence(dataset.X, "cat_small", "cat_with_nan")
        self.assertEqual(
            result.loc["category_cooccurrence", "cat_small__cat_with_nan"],
            expected_all,
        )

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=27)
        result = self.compute_statistic(CategoryCooccurrenceStatistic, dataset)

        self.assertTrue(result.empty)
