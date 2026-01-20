"""Unit tests for KurtosisStatistic."""
from __future__ import annotations

import pandas as pd

from iaml.data_type import DataType
from iaml.statistics.kurtosis import KurtosisStatistic

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)
from tests.statistics.statistic_test_case import StatisticTestCase


class TestKurtosisStatistic(StatisticTestCase):
    """Tests for KurtosisStatistic."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def _assert_kurtosis_value(self, result, column: str, expected) -> None:
        if pd.isna(expected):
            self.assertTrue(pd.isna(result.loc["kurtosis", column]))
        else:
            self.assertAlmostEqual(result.loc["kurtosis", column], expected)

    def test_classification_kurtosis_columns_and_values(self) -> None:
        X, y = make_statistics_classification_data(n_samples=18, seed=22, n_classes=3)
        dataset = self.make_dataset(X, y)
        statistic = KurtosisStatistic()
        result = statistic.compute(dataset)

        class_labels = list(pd.unique(dataset.y))
        expected_columns = [
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        ]
        self.assertCountEqual(list(result.columns), expected_columns)
        self.assertListEqual(list(result.index), ["kurtosis"])

        all_expected = dataset.X["num_2"].kurtosis()
        self._assert_kurtosis_value(result, "num_2_all", all_expected)

        sample_label = class_labels[0]
        label_expected = dataset.X.loc[dataset.y == sample_label, "num_2"].kurtosis()
        self._assert_kurtosis_value(result, f"num_2_{sample_label}", label_expected)

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["kurtosis", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["kurtosis", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_returns_empty_dataframe(self) -> None:
        X, y = make_statistics_regression_data(n_samples=12, seed=32)
        dataset = self.make_dataset(X, y)
        statistic = KurtosisStatistic()
        result = statistic.compute(dataset)
        self.assertTrue(result.empty)

    def test_survival_returns_empty_dataframe(self) -> None:
        X, y = make_statistics_survival_data(n_samples=12, seed=42)
        dataset = self.make_dataset(X, y)
        statistic = KurtosisStatistic()
        result = statistic.compute(dataset)
        self.assertTrue(result.empty)
