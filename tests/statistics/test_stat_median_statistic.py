"""Unit tests for MedianStatistic."""
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
from tests.statistics.statistic_test_case import StatisticTestCase

MedianStatistic = _load_module(
    "iaml.statistics.median_statistic",
    IAML_PATH / "statistics" / r"median_statistic.py",
).MedianStatistic


class TestMedianStatistic(StatisticTestCase):
    """Coverage for MedianStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def test_classification_medians_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=23, n_classes=3)
        result = self.compute_statistic(MedianStatistic, dataset)

        self.assertEqual(set(result.index), {"median"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        all_expected = dataset.X["num_2"].median()
        self.assertAlmostEqual(result.loc["median", "num_2_all"], all_expected)

        sample_label = class_labels[0]
        label_expected = dataset.X.loc[dataset.y == sample_label, "num_2"].median()
        self.assertAlmostEqual(
            result.loc["median", f"num_2_{sample_label}"],
            label_expected,
        )

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["median", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["median", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_medians_exclude_text_columns(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=33)
        result = self.compute_statistic(MedianStatistic, dataset)

        self.assertEqual(set(result.index), {"median"})
        self.assertEqual(set(result.columns), set(self._non_text_columns(dataset)))

        all_expected = dataset.X["num_2"].median()
        self.assertAlmostEqual(result.loc["median", "num_2"], all_expected)
        self.assertTrue(pd.isna(result.loc["median", "cat_small"]))
        self.assertTrue(pd.isna(result.loc["median", "date_col"]))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=10, seed=43)
        result = self.compute_statistic(MedianStatistic, dataset)

        self.assertTrue(result.empty)
