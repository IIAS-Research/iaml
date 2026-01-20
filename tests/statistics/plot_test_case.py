"""Base helpers for statistic plot tests."""
from __future__ import annotations

from typing import Type
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running tests from the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from iaml.plot import StatisticPlot

from .statistic_test_case import StatisticTestCase


class StatisticPlotTestCase(StatisticTestCase):
    """Shared helpers to make writing statistic plot tests concise."""

    def compute_plot(
        self,
        plot_cls: Type[StatisticPlot],
        stats_df: pd.DataFrame,
        **kwargs,
    ) -> StatisticPlot:
        plot = plot_cls()
        plot.compute(stats_df, **kwargs)
        return plot

    def assert_plot_image(self, plot: StatisticPlot) -> None:
        image = plot.image
        self.assertIsInstance(image, (bytes, bytearray))
        self.assertGreater(len(image), 0)

    def group_columns_by_feature(self, dataframe: pd.DataFrame) -> dict[str, list[str]]:
        columns = list(dataframe.columns)
        base_names: list[str] = []
        for col in columns:
            if isinstance(col, str) and col.endswith('_all'):
                base = col[:-4]
                if base not in base_names:
                    base_names.append(base)

        groups: dict[str, list[str]] = {}
        used_cols: set[str] = set()
        if base_names:
            for base in base_names:
                group = [
                    col for col in columns
                    if col == base or (isinstance(col, str) and col.startswith(f"{base}_"))
                ]
                groups[base] = group
                used_cols.update(group)

        for col in columns:
            if col not in used_cols:
                groups[str(col)] = [col]

        return groups
