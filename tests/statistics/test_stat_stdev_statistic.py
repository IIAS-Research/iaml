"""Unit tests for StdevStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.stdev import StdevStatistic
from tests.statistics.statistic_test_case import StatisticTestCase


class TestStdevStatistic(StatisticTestCase):
    """Coverage for StdevStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def test_classification_stdevs_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=27, n_classes=3)
        result = self.compute_statistic(StdevStatistic, dataset)

        self.assertEqual(set(result.index), {"stdev"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        all_expected = dataset.X["num_2"].std()
        self.assertAlmostEqual(result.loc["stdev", "num_2_all"], all_expected)

        sample_label = class_labels[0]
        label_expected = dataset.X.loc[dataset.y == sample_label, "num_2"].std()
        self.assertAlmostEqual(result.loc["stdev", f"num_2_{sample_label}"], label_expected)

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["stdev", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["stdev", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_non_classification_returns_empty(self) -> None:
        datasets = [
            ("continuous", self.make_regression_dataset(n_samples=12, seed=31)),
            ("survival", self.make_survival_dataset(n_samples=12, seed=41)),
        ]

        for name, dataset in datasets:
            with self.subTest(name=name):
                result = self.compute_statistic(StdevStatistic, dataset)
                self.assertTrue(result.empty)
