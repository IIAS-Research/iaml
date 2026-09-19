"""[STATISTIC] Time By Group."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class TimeByGroupStatistic(Statistic):
    """[STATISTIC] Time By Group."""

    name: str = "Time By Group"
    _description: str = textwrap.dedent("""\
        Time by group reports event rate and median time by category.
        """)
    _description_long: str = textwrap.dedent("""\
        Time by group reports event rate and median survival time per category
        for categorical features in survival targets.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'time_by_group'

    def _select_columns(self, dataset: Dataset) -> list[str]:
        columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        category_columns = list(dataset.X.select_dtypes(include=['category']).columns)
        seen = set()
        ordered = []
        for column in columns + category_columns:
            if column in dataset.X.columns and column not in seen:
                ordered.append(column)
                seen.add(column)
        return ordered

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute event rate and median time by categorical group."""
        if dataset.type_of_target != 'survival':
            return pd.DataFrame()

        columns = self._select_columns(dataset)
        if not columns:
            return pd.DataFrame()

        samples = Dataset.normalize_survival_target(dataset.y)
        if not samples or len(samples) != len(dataset.X):
            return pd.DataFrame()

        events = np.asarray([event for event, _ in samples], dtype=float)
        times = np.asarray([time for _, time in samples], dtype=float)

        frame = dataset.X[columns].copy()
        frame['_event'] = events
        frame['_time'] = times

        data = []
        df_columns = []
        for col in columns:
            values = frame[[col, '_event', '_time']].dropna(subset=[col])
            if values.empty:
                continue
            grouped = values.groupby(col, sort=True)
            event_rates = grouped['_event'].mean()
            median_times = grouped['_time'].median()
            for group_value in event_rates.index:
                value_label = str(group_value)
                df_columns.append(f"{col}_{value_label}_event_rate")
                data.append(float(event_rates.loc[group_value]))
                df_columns.append(f"{col}_{value_label}_median_time")
                data.append(float(median_times.loc[group_value]))

        if not df_columns:
            return pd.DataFrame()

        return pd.DataFrame([data], index=[str(self)], columns=df_columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target == 'survival'
