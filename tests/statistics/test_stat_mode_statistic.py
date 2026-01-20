"""Unit tests for ModeStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.mode import ModeStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestModeStatistic(StatisticTestCase):
    """Coverage for ModeStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def test_classification_columns_exclude_text(self) -> None:
        dataset = self.make_classification_dataset(n_samples=24, seed=21, n_classes=3)
        result = self.compute_statistic(ModeStatistic, dataset)

        self.assertEqual(set(result.index), {"mode"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        for label in ["all"] + class_labels:
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_classification_mode_values_per_class(self) -> None:
        X = pd.DataFrame(
            {
                "num_1": [1, 1, 2, 2],
                "cat_small": ["a", "a", "b", "b"],
                "date_col": pd.to_datetime(
                    ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
                ),
            }
        )
        y = ["class_a", "class_a", "class_b", "class_b"]
        dataset = self.make_dataset(X, y)
        result = self.compute_statistic(ModeStatistic, dataset)

        class_labels = list(pd.unique(dataset.y))
        for col in self._non_text_columns(dataset):
            all_expected = X[col].mode().to_list()
            self.assertEqual(result.loc["mode", f"{col}_all"], all_expected)
            for label in class_labels:
                expected = X.loc[pd.Series(y) == label, col].mode().to_list()
                self.assertEqual(result.loc["mode", f"{col}_{label}"], expected)

    def test_non_classification_returns_empty(self) -> None:
        regression = self.make_regression_dataset(n_samples=12, seed=33)
        survival = self.make_survival_dataset(n_samples=12, seed=41)

        self.assertTrue(self.compute_statistic(ModeStatistic, regression).empty)
        self.assertTrue(self.compute_statistic(ModeStatistic, survival).empty)
