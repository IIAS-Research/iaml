"""[STATISTIC] Rare Category Rate."""
from __future__ import annotations

import math
import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class RareCategoryRateStatistic(Statistic):
    """[STATISTIC] Rare Category Rate."""

    name: str = "Rare Category Rate"
    _description: str = textwrap.dedent("""\
        Rare category rate measures the share of rare categories in categorical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Rare category rate measures the ratio of categories whose frequency is below a
        threshold for categorical columns, optionally per class.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'rare_category_rate'

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

    def _rare_ratio(self, series: pd.Series, threshold: float) -> float:
        counts = series.value_counts(dropna=True)
        total = int(counts.sum())
        if total <= 0 or counts.empty:
            return 0.0
        if threshold <= 0:
            return 0.0
        ratios = counts / total
        rare_count = int((ratios < threshold).sum())
        return rare_count / len(counts)

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute the rare category rate for categorical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        threshold = kwargs.get('threshold')
        if threshold is None:
            threshold = kwargs.get('min_frequency', 0.01)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.01
        if not math.isfinite(threshold):
            threshold = 0.01

        columns = self._select_columns(dataset)
        if not columns:
            return pd.DataFrame()

        data = []
        df_columns = []

        if dataset.type_of_target == 'continuous':
            for col in columns:
                df_columns.append(col)
                data.append(self._rare_ratio(dataset.X[col], threshold))
            return pd.DataFrame([data], index=[str(self)], columns=df_columns)

        class_labels = list(pd.unique(dataset.y))
        for col in columns:
            for label in ['all'] + class_labels:
                df_columns.append(f"{col}_{label}")
                if label == 'all':
                    values = dataset.X[col]
                else:
                    values = dataset.X.loc[dataset.y == label][col]
                data.append(self._rare_ratio(values, threshold))
        return pd.DataFrame([data], index=[str(self)], columns=df_columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
