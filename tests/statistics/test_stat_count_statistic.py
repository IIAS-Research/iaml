"""Unit tests for CountStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.statistics.count import CountStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestCountStatistic(StatisticTestCase):
    """Coverage for CountStatistic core behavior."""

    def test_classification_columns_and_null_counts(self) -> None:
        dataset = self.make_classification_dataset(n_samples=40, seed=7, n_classes=3)
        result = self.compute_statistic(CountStatistic, dataset)

        self.assertEqual(set(result.index), {"count", "null_count"})

        class_labels = list(pd.unique(dataset.y))
        excluded = {"short_text", "long_text"}
        expected_columns = set()
        for col in dataset.X.columns:
            if col in excluded:
                continue
            for label in ["all"] + class_labels:
                expected_columns.add(f"{col}_{label}")

        self.assertEqual(set(result.columns), expected_columns)
        self.assertTrue((result.loc["null_count"] <= result.loc["count"]).all())

    def test_continuous_counts_match_nan_handling(self) -> None:
        dataset = self.make_regression_dataset(n_samples=50, seed=11)
        result = self.compute_statistic(CountStatistic, dataset)

        self.assertEqual(set(result.index), {"count", "null_count"})
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        expected_count = dataset.X["num_2"].count()
        expected_nulls = dataset.X["num_2"].isna().sum()
        self.assertEqual(result.loc["count", "num_2"], expected_count)
        self.assertEqual(result.loc["null_count", "num_2"], expected_nulls)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=30, seed=13)
        result = self.compute_statistic(CountStatistic, dataset)

        self.assertTrue(result.empty)
