"""Unit tests for ValueCountsStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.value_counts import ValueCountsStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestValueCountsStatistic(StatisticTestCase):
    """Coverage for ValueCountsStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def _categorical_columns(self, dataset) -> list[str]:
        return [
            col
            for col in self._non_text_columns(dataset)
            if dataset.columns_types[col][1] == DataType.CATEGORICAL
        ]

    def test_classification_value_counts_per_class(self) -> None:
        dataset = self.make_classification_dataset(n_samples=36, seed=5, n_classes=3)
        result = self.compute_statistic(ValueCountsStatistic, dataset)

        self.assertEqual(set(result.index), {"value_counts"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        for col in self._categorical_columns(dataset):
            all_expected = list(dataset.X[col].value_counts().items())
            self.assertEqual(result.loc["value_counts", f"{col}_all"], all_expected)
            for label in class_labels:
                expected = list(
                    dataset.X.loc[dataset.y == label, col].value_counts().items()
                )
                self.assertEqual(
                    result.loc["value_counts", f"{col}_{label}"],
                    expected,
                )

        for col in self._non_text_columns(dataset):
            if col in self._categorical_columns(dataset):
                continue
            for label in ["all"] + class_labels:
                self.assertIsNone(result.loc["value_counts", f"{col}_{label}"])

    def test_continuous_value_counts_exclude_text(self) -> None:
        dataset = self.make_regression_dataset(n_samples=40, seed=17)
        result = self.compute_statistic(ValueCountsStatistic, dataset)

        self.assertEqual(set(result.index), {"value_counts"})
        self.assertEqual(set(result.columns), set(self._non_text_columns(dataset)))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        for col in self._categorical_columns(dataset):
            expected = list(dataset.X[col].value_counts().items())
            self.assertEqual(result.loc["value_counts", col], expected)

        for col in self._non_text_columns(dataset):
            if col in self._categorical_columns(dataset):
                continue
            self.assertIsNone(result.loc["value_counts", col])

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=24, seed=19)
        result = self.compute_statistic(ValueCountsStatistic, dataset)

        self.assertTrue(result.empty)
