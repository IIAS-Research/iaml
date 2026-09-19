"""[STATISTIC] Unique Count."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class UniqueCountStatistic(Statistic):
    """[STATISTIC] Unique Count."""

    name: str = "Unique Count"
    _description: str = textwrap.dedent("""\
        Unique count measures the number of distinct non-null values per column.
        """)
    _description_long: str = textwrap.dedent("""\
        Unique count measures the number of distinct non-null values per column,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'nunique'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute unique counts for each column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    data.append(dataset.X[col].nunique(dropna=True))
            return pd.DataFrame([data], index=[str(self)], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if label == 'all':
                        values = dataset.X[col]
                    else:
                        values = dataset.X.loc[dataset.y == label][col]
                    data.append(values.nunique(dropna=True))
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
