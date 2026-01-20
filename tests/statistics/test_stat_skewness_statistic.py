"""Unit tests for SkewnessStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.skewness import SkewnessStatistic

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)
from tests.statistics.statistic_test_case import StatisticTestCase


class TestSkewnessStatistic(StatisticTestCase):
    """Tests for SkewnessStatistic."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def _assert_skewness_value(self, result, column: str, expected) -> None:
        if pd.isna(expected):
            self.assertTrue(pd.isna(result.loc["skewness", column]))
        else:
            self.assertAlmostEqual(result.loc["skewness", column], expected)

    def test_classification_skewness_columns_and_values(self) -> None:
        X, y = make_statistics_classification_data(n_samples=18, seed=21, n_classes=3)
        dataset = self.make_dataset(X, y)
        statistic = SkewnessStatistic()
        result = statistic.compute(dataset)

        class_labels = list(pd.unique(dataset.y))
        expected_columns = [
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        ]
        self.assertCountEqual(list(result.columns), expected_columns)
        self.assertListEqual(list(result.index), ["skewness"])

        all_expected = dataset.X["num_2"].skew()
        self._assert_skewness_value(result, "num_2_all", all_expected)

        sample_label = class_labels[0]
        label_expected = dataset.X.loc[dataset.y == sample_label, "num_2"].skew()
        self._assert_skewness_value(result, f"num_2_{sample_label}", label_expected)

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["skewness", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["skewness", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_returns_empty_dataframe(self) -> None:
        X, y = make_statistics_regression_data(n_samples=12, seed=31)
        dataset = self.make_dataset(X, y)
        statistic = SkewnessStatistic()
        result = statistic.compute(dataset)
        self.assertTrue(result.empty)

    def test_survival_returns_empty_dataframe(self) -> None:
        X, y = make_statistics_survival_data(n_samples=12, seed=41)
        dataset = self.make_dataset(X, y)
        statistic = SkewnessStatistic()
        result = statistic.compute(dataset)
        self.assertTrue(result.empty)
