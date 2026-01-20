"""Unit tests for BoundStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.minmax import BoundStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestBoundStatistic(StatisticTestCase):
    """Coverage for BoundStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def test_classification_min_max_values_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=21, n_classes=3)
        result = self.compute_statistic(BoundStatistic, dataset)

        self.assertEqual(set(result.index), {"min", "max"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        expected_min = dataset.X["num_2"].min()
        expected_max = dataset.X["num_2"].max()
        self.assertAlmostEqual(result.loc["min", "num_2_all"], expected_min)
        self.assertAlmostEqual(result.loc["max", "num_2_all"], expected_max)

        sample_label = class_labels[0]
        label_mask = dataset.y == sample_label
        label_min = dataset.X.loc[label_mask, "num_2"].min()
        label_max = dataset.X.loc[label_mask, "num_2"].max()
        self.assertAlmostEqual(result.loc["min", f"num_2_{sample_label}"], label_min)
        self.assertAlmostEqual(result.loc["max", f"num_2_{sample_label}"], label_max)

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["min", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["max", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["min", f"date_col_{label}"]))
            self.assertTrue(pd.isna(result.loc["max", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_returns_empty(self) -> None:
        dataset = self.make_regression_dataset(n_samples=12, seed=31)
        result = self.compute_statistic(BoundStatistic, dataset)

        self.assertTrue(result.empty)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=10, seed=41)
        result = self.compute_statistic(BoundStatistic, dataset)

        self.assertTrue(result.empty)
