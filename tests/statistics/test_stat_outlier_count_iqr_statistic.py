"""Unit tests for OutlierCountIQRStatistic."""
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

OutlierCountIQRStatistic = _load_module(
    "iaml.statistics.outlier_count_iqr_statistic",
    IAML_PATH / "statistics" / "outlier_count_iqr_statistic.py",
).OutlierCountIQRStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


class TestOutlierCountIQRStatistic(StatisticTestCase):
    """Coverage for OutlierCountIQRStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def _count_outliers(self, values: pd.Series) -> int:
        values = values.dropna()
        if values.empty:
            return 0
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr):
            return 0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((values < lower) | (values > upper)).sum())

    def test_classification_outlier_counts_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=30, seed=21, n_classes=3)
        result = self.compute_statistic(OutlierCountIQRStatistic, dataset)

        self.assertEqual(list(result.index), ["outlier_count_iqr"])

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        all_expected = self._count_outliers(dataset.X["num_2"])
        self.assertEqual(result.loc["outlier_count_iqr", "num_2_all"], all_expected)

        sample_label = class_labels[0]
        label_values = dataset.X.loc[dataset.y == sample_label, "num_2"]
        label_expected = self._count_outliers(label_values)
        self.assertEqual(
            result.loc["outlier_count_iqr", f"num_2_{sample_label}"], label_expected
        )

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["outlier_count_iqr", f"cat_small_{label}"]))
            self.assertTrue(
                pd.isna(result.loc["outlier_count_iqr", f"cat_with_nan_{label}"])
            )
            self.assertTrue(pd.isna(result.loc["outlier_count_iqr", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_outlier_counts_exclude_text(self) -> None:
        dataset = self.make_regression_dataset(n_samples=26, seed=37)
        result = self.compute_statistic(OutlierCountIQRStatistic, dataset)

        self.assertEqual(list(result.index), ["outlier_count_iqr"])
        self.assertEqual(set(result.columns), set(self._non_text_columns(dataset)))

        expected = self._count_outliers(dataset.X["num_1"])
        self.assertEqual(result.loc["outlier_count_iqr", "num_1"], expected)
        self.assertTrue(pd.isna(result.loc["outlier_count_iqr", "cat_small"]))
        self.assertTrue(pd.isna(result.loc["outlier_count_iqr", "cat_with_nan"]))
        self.assertTrue(pd.isna(result.loc["outlier_count_iqr", "date_col"]))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=41)
        result = self.compute_statistic(OutlierCountIQRStatistic, dataset)

        self.assertTrue(result.empty)
