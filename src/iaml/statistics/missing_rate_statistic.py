"""[STATISTIC] Missing Rate."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class MissingRateStatistic(Statistic):
    """[STATISTIC] Missing Rate."""

    name: str = "Missing Rate"
    _description: str = textwrap.dedent("""\
        Missing rate measures the percentage of missing values per column.
        """)
    _description_long: str = textwrap.dedent("""\
        Missing rate measures the percentage of missing values per column,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'missing_rate'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute missing rate for each column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    values = dataset.X[col]
                    n_rows = int(values.shape[0])
                    rate = values.isna().mean() if n_rows else 0.0
                    data.append(rate)
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
                    n_rows = int(values.shape[0])
                    rate = values.isna().mean() if n_rows else 0.0
                    data.append(rate)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
