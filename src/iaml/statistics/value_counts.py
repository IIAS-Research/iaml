"""[STATISTIC] Value counts."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class ValueCountsStatistic(Statistic):
    """[STATISTIC] Value counts."""

    name: str = "Value Counts"
    _description: str = textwrap.dedent("""\
        Value counts measures value frequencies for categorical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Value counts measures value frequencies for categorical columns, optionally per class.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'value_counts'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute value counts for categorical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                        counts = dataset.X[col].value_counts()
                        data.append(list((idx, val) for idx, val in counts.items()))
                    else:
                        data.append(None)
            return pd.DataFrame([data], index=[str(self)], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                        if label == 'all':
                            counts = dataset.X[col].value_counts()
                        else:
                            counts = dataset.X.loc[dataset.y == label][col].value_counts()
                        data.append(list((idx, val) for idx, val in counts.items()))
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
