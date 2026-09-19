"""Unit tests for LinePlot."""
from __future__ import annotations

import pandas as pd

from iaml.plots import LinePlot
from tests.statistics.contracts import build_statistics_dataframe
from tests.statistics.plot_test_case import StatisticPlotTestCase


class TestLinePlot(StatisticPlotTestCase):
    """Coverage for LinePlot core behavior."""

    def _feature_stats(self, stats_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        groups = self.group_columns_by_feature(stats_df)
        return stats_df[groups[feature]]

    def test_classification_line_plot(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=21, n_classes=3)
        stats_df = build_statistics_dataframe(dataset)
        feature_stats = self._feature_stats(stats_df, "cat_small")

        self.assertIn("count", feature_stats.index)
        self.assertIn("null_count", feature_stats.index)
        plot = self.compute_plot(LinePlot, feature_stats)

        self.assert_plot_image(plot)

    def test_regression_line_plot(self) -> None:
        dataset = self.make_regression_dataset(n_samples=16, seed=31)
        stats_df = build_statistics_dataframe(dataset)
        feature_stats = self._feature_stats(stats_df, "num_1")

        self.assertIn("count", feature_stats.index)
        self.assertIn("null_count", feature_stats.index)
        plot = self.compute_plot(LinePlot, feature_stats)

        self.assert_plot_image(plot)

    def test_missing_required_rows_placeholder(self) -> None:
        stats_df = pd.DataFrame({"num_1": [12]}, index=["count"])
        plot = self.compute_plot(LinePlot, stats_df)

        self.assert_plot_image(plot)
