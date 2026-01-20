"""Unit tests for BarPlot."""
from __future__ import annotations

import pandas as pd

from iaml.plots import BarPlot
from tests.statistics.contracts import build_statistics_dataframe
from tests.statistics.plot_test_case import StatisticPlotTestCase


class TestBarPlot(StatisticPlotTestCase):
    """Coverage for BarPlot core behavior."""

    def _feature_stats(self, stats_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        groups = self.group_columns_by_feature(stats_df)
        return stats_df[groups[feature]]

    def test_classification_categorical_bar_plot(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=21, n_classes=3)
        stats_df = build_statistics_dataframe(dataset)
        feature_stats = self._feature_stats(stats_df, "cat_small")

        self.assertIn("value_counts", feature_stats.index)
        plot = self.compute_plot(BarPlot, feature_stats)

        self.assert_plot_image(plot)

    def test_continuous_numeric_bar_plot(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=31)
        stats_df = build_statistics_dataframe(dataset)
        feature_stats = self._feature_stats(stats_df, "num_1")

        self.assertIn("mean", feature_stats.index)
        plot = self.compute_plot(BarPlot, feature_stats)

        self.assert_plot_image(plot)

    def test_empty_dataframe_placeholder(self) -> None:
        plot = self.compute_plot(BarPlot, pd.DataFrame())

        self.assert_plot_image(plot)
