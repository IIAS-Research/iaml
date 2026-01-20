"""Unit tests for ViolinStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.violin import ViolinStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestViolinStatistic(StatisticTestCase):
    """Coverage for ViolinStatistic core behavior."""

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

    def test_continuous_violin_shapes_and_columns(self) -> None:
        dataset = self.make_regression_dataset(n_samples=60, seed=12)
        result = self.compute_statistic(ViolinStatistic, dataset)

        self.assertEqual(set(result.index), {"violin"})
        self.assertEqual(set(result.columns), set(self._non_text_columns(dataset)))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

        for col in self._categorical_columns(dataset):
            stats_dict = result.loc["violin", col]
            self.assertIsInstance(stats_dict, dict)
            self.assertTrue(stats_dict)
            for stats in stats_dict.values():
                self.assertEqual(set(stats.keys()), {"density", "support", "quartiles"})
                self.assertEqual(len(stats["density"]), 100)
                self.assertEqual(len(stats["support"]), 100)
                self.assertEqual(len(stats["quartiles"]), 3)

        for col in self._non_text_columns(dataset):
            if col in self._categorical_columns(dataset):
                continue
            self.assertIsNone(result.loc["violin", col])

    def test_singleton_category_is_ignored(self) -> None:
        X = pd.DataFrame(
            {
                "cat": ["A", "A", "B"],
                "num": [1.0, 2.0, 3.0],
            }
        )
        y = [0.1, 0.2, 0.3]
        dataset = self.make_dataset(X, y=y)
        result = self.compute_statistic(ViolinStatistic, dataset)

        stats_dict = result.loc["violin", "cat"]
        self.assertIn("A", stats_dict)
        self.assertNotIn("B", stats_dict)
        self.assertIsNone(result.loc["violin", "num"])

    def test_non_continuous_returns_empty(self) -> None:
        dataset = self.make_classification_dataset(n_samples=24, seed=17)
        result = self.compute_statistic(ViolinStatistic, dataset)

        self.assertTrue(result.empty)
