"""[STATISTIC] Most Frequent Ratio."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class MostFrequentRatioStatistic(Statistic):
    """[STATISTIC] Most Frequent Ratio."""

    name: str = "Most Frequent Ratio"
    _description: str = textwrap.dedent("""\
        Most frequent ratio measures the share of the dominant category in
        categorical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Most frequent ratio measures the ratio of the most common category in
        categorical columns, optionally per class.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'most_frequent_ratio'

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

    def _most_frequent_ratio(self, series: pd.Series) -> float:
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return 0.0
        total = int(counts.sum())
        if total <= 0:
            return 0.0
        top = int(counts.max())
        return top / total

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute most frequent ratio for categorical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        columns = self._select_columns(dataset)
        if not columns:
            return pd.DataFrame()

        data = []
        df_columns = []

        if dataset.type_of_target == 'continuous':
            for col in columns:
                df_columns.append(col)
                data.append(self._most_frequent_ratio(dataset.X[col]))
            return pd.DataFrame([data], index=[str(self)], columns=df_columns)

        class_labels = list(pd.unique(dataset.y))
        for col in columns:
            for label in ['all'] + class_labels:
                df_columns.append(f"{col}_{label}")
                if label == 'all':
                    values = dataset.X[col]
                else:
                    values = dataset.X.loc[dataset.y == label][col]
                data.append(self._most_frequent_ratio(values))
        return pd.DataFrame([data], index=[str(self)], columns=df_columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
