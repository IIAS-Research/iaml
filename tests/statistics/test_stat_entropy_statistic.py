"""Unit tests for EntropyStatistic."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
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
from tests.statistics.statistic_test_case import StatisticTestCase

EntropyStatistic = _load_module(
    "iaml.statistics.entropy_statistic",
    IAML_PATH / "statistics" / r"entropy_statistic.py",
).EntropyStatistic


class TestEntropyStatistic(StatisticTestCase):
    """Coverage for EntropyStatistic core behavior."""

    def _categorical_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] == DataType.CATEGORICAL
        ]

    def _entropy(self, series: pd.Series) -> float:
        values = series.dropna()
        if values.empty:
            return 0.0
        counts = values.value_counts()
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        return float(-(probs * np.log2(probs)).sum())

    def test_classification_entropy_per_class(self) -> None:
        dataset = self.make_classification_dataset(n_samples=30, seed=7, n_classes=3)
        result = self.compute_statistic(EntropyStatistic, dataset)

        self.assertEqual(set(result.index), {"entropy"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._categorical_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        for col in self._categorical_columns(dataset):
            all_expected = self._entropy(dataset.X[col])
            self.assertAlmostEqual(result.loc["entropy", f"{col}_all"], all_expected)
            for label in class_labels:
                label_expected = self._entropy(dataset.X.loc[dataset.y == label, col])
                self.assertAlmostEqual(
                    result.loc["entropy", f"{col}_{label}"],
                    label_expected,
                )

    def test_continuous_entropy_categorical_only(self) -> None:
        dataset = self.make_regression_dataset(n_samples=24, seed=11)
        result = self.compute_statistic(EntropyStatistic, dataset)

        self.assertEqual(set(result.index), {"entropy"})
        self.assertEqual(set(result.columns), set(self._categorical_columns(dataset)))
        self.assertNotIn("num_1", result.columns)
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        for col in self._categorical_columns(dataset):
            expected = self._entropy(dataset.X[col])
            self.assertAlmostEqual(result.loc["entropy", col], expected)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=13)
        result = self.compute_statistic(EntropyStatistic, dataset)

        self.assertTrue(result.empty)
