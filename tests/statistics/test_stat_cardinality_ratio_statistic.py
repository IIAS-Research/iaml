"""Unit tests for CardinalityRatioStatistic."""
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

from tests.statistics.statistic_test_case import StatisticTestCase

_load_module("iaml.cache_keys", IAML_PATH / "cache_keys.py")
_load_module("iaml.data_type", IAML_PATH / "data_type.py")
_load_module("iaml.type_of_target", IAML_PATH / "type_of_target.py")
_load_module("iaml.logger", IAML_PATH / "logger.py")
_load_module("iaml.reference", IAML_PATH / "reference.py")
_load_module("iaml.dataset", IAML_PATH / "dataset.py")
_load_module("iaml.statistic", IAML_PATH / "statistic.py")

DataType = sys.modules["iaml.data_type"].DataType

CardinalityRatioStatistic = _load_module(
    "iaml.statistics.cardinality_ratio_statistic",
    IAML_PATH / "statistics" / "cardinality_ratio_statistic.py",
).CardinalityRatioStatistic


class TestCardinalityRatioStatistic(StatisticTestCase):
    """Coverage for CardinalityRatioStatistic core behavior."""

    def test_classification_ratios_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=24, seed=5, n_classes=3)
        result = self.compute_statistic(CardinalityRatioStatistic, dataset)

        self.assertEqual(list(result.index), ["cardinality_ratio"])

        non_text_columns = [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]
        text_columns = [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] in (DataType.TEXT, DataType.SHORT_TEXT)
        ]
        self.assertTrue(non_text_columns)
        self.assertTrue(text_columns)

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in non_text_columns
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        sample_col = non_text_columns[0]
        expected_all = dataset.X[sample_col].nunique(dropna=True) / len(dataset.X)
        self.assertAlmostEqual(
            result.loc["cardinality_ratio", f"{sample_col}_all"],
            expected_all,
        )
        for label in class_labels:
            values = dataset.X.loc[dataset.y == label, sample_col]
            expected = values.nunique(dropna=True) / len(values) if len(values) else 0.0
            self.assertAlmostEqual(
                result.loc["cardinality_ratio", f"{sample_col}_{label}"],
                expected,
            )

        for col in text_columns:
            self.assertFalse(any(name.startswith(f"{col}_") for name in result.columns))

    def test_continuous_ratios_exclude_text(self) -> None:
        dataset = self.make_regression_dataset(n_samples=20, seed=7)
        result = self.compute_statistic(CardinalityRatioStatistic, dataset)

        self.assertEqual(list(result.index), ["cardinality_ratio"])
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)
        self.assertIn("cat_small", result.columns)

        expected_ratio = dataset.X["num_2"].nunique(dropna=True) / len(dataset.X)
        self.assertAlmostEqual(result.loc["cardinality_ratio", "num_2"], expected_ratio)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=15)
        result = self.compute_statistic(CardinalityRatioStatistic, dataset)

        self.assertTrue(result.empty)
