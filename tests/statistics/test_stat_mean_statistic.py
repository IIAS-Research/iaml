"""Unit tests for MeanStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.mean import MeanStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestMeanStatistic(StatisticTestCase):
    """Coverage for MeanStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def test_classification_means_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=23, n_classes=3)
        result = self.compute_statistic(MeanStatistic, dataset)

        self.assertEqual(set(result.index), {"mean"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        all_expected = dataset.X["num_2"].mean()
        self.assertAlmostEqual(result.loc["mean", "num_2_all"], all_expected)

        sample_label = class_labels[0]
        label_expected = dataset.X.loc[dataset.y == sample_label, "num_2"].mean()
        self.assertAlmostEqual(result.loc["mean", f"num_2_{sample_label}"], label_expected)

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["mean", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["mean", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_means_exclude_text_columns(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=33)
        result = self.compute_statistic(MeanStatistic, dataset)

        self.assertEqual(set(result.index), {"mean"})
        self.assertEqual(set(result.columns), set(self._non_text_columns(dataset)))

        all_expected = dataset.X["num_2"].mean()
        self.assertAlmostEqual(result.loc["mean", "num_2"], all_expected)
        self.assertTrue(pd.isna(result.loc["mean", "cat_small"]))
        self.assertTrue(pd.isna(result.loc["mean", "date_col"]))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=10, seed=43)
        result = self.compute_statistic(MeanStatistic, dataset)

        self.assertTrue(result.empty)
