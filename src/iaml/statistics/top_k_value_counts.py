"""[STATISTIC] Top-K Value Counts."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class TopKValueCountsStatistic(Statistic):
    """[STATISTIC] Top-K Value Counts."""

    name: str = "Top-K Value Counts"
    _description: str = textwrap.dedent("""\
        Top-k value counts list the most frequent categories and their shares.
        """)
    _description_long: str = textwrap.dedent("""\
        Top-k value counts list the most frequent categories and their shares for
        categorical columns, optionally per class.
        """)
    refs: list[dict] = []

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def __str__(self) -> str:
        return f'top_{self.k}_value_counts'

    def _top_k_counts(self, series: pd.Series) -> list[tuple[object, int, float]]:
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return []
        top_counts = counts.head(self.k)
        total = counts.sum()
        return [(idx, int(val), float(val) / total) for idx, val in top_counts.items()]

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute top-k value counts for categorical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                    columns.append(col)
                    data.append(self._top_k_counts(dataset.X[col]))
            return pd.DataFrame([data], index=[str(self)], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if label == 'all':
                        values = dataset.X[col]
                    else:
                        values = dataset.X.loc[dataset.y == label][col]
                    data.append(self._top_k_counts(values))
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
