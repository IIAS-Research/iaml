"""Unit tests for UniqueCountStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)


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

DataType = sys.modules["iaml.data_type"].DataType
Dataset = sys.modules["iaml.dataset"].Dataset

UniqueCountStatistic = _load_module(
    "iaml.statistics.unique_count_statistic",
    IAML_PATH / "statistics" / "unique_count_statistic.py",
).UniqueCountStatistic


class TestUniqueCountStatistic(unittest.TestCase):
    """Coverage for UniqueCountStatistic core behavior."""

    def _make_classification_dataset(
        self,
        n_samples: int = 24,
        seed: int = 5,
        n_classes: int = 3,
    ) -> Dataset:
        X, y = make_statistics_classification_data(
            n_samples=n_samples,
            seed=seed,
            n_classes=n_classes,
        )
        return Dataset(X, y)

    def _make_regression_dataset(
        self,
        n_samples: int = 30,
        seed: int = 9,
    ) -> Dataset:
        X, y = make_statistics_regression_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def _make_survival_dataset(
        self,
        n_samples: int = 12,
        seed: int = 17,
    ) -> Dataset:
        X, y = make_statistics_survival_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def _compute_statistic(self, dataset: Dataset) -> pd.DataFrame:
        statistic = UniqueCountStatistic()
        return statistic.compute(dataset)

    def test_classification_columns_and_counts(self) -> None:
        dataset = self._make_classification_dataset()
        result = self._compute_statistic(dataset)

        self.assertEqual(list(result.index), ["nunique"])

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
        expected_all = dataset.X[sample_col].nunique(dropna=True)
        self.assertEqual(result.loc["nunique", f"{sample_col}_all"], expected_all)
        for label in class_labels:
            expected = dataset.X.loc[dataset.y == label, sample_col].nunique(dropna=True)
            self.assertEqual(result.loc["nunique", f"{sample_col}_{label}"], expected)

        for col in text_columns:
            self.assertFalse(any(name.startswith(f"{col}_") for name in result.columns))

    def test_continuous_counts_exclude_text(self) -> None:
        dataset = self._make_regression_dataset()
        result = self._compute_statistic(dataset)

        self.assertEqual(list(result.index), ["nunique"])
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        expected_num2 = dataset.X["num_2"].nunique(dropna=True)
        self.assertEqual(result.loc["nunique", "num_2"], expected_num2)

    def test_survival_returns_empty(self) -> None:
        dataset = self._make_survival_dataset()
        result = self._compute_statistic(dataset)

        self.assertTrue(result.empty)
