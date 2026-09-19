"""[STATISTIC] Count."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class CountStatistic(Statistic):
    """[STATISTIC] Count."""

    name: str = "Count"
    _description: str = textwrap.dedent("""\
        Count measures the number of non-null values for each column.
        """)
    _description_long: str = textwrap.dedent("""\
        Count measures the number of non-null values (or null values when requested)
        for each column.
        """)
    refs: list[dict] = []

    count_to_compute = [True, False]

    def __init__(self, null_count: bool = False) -> None:
        self.null_count = null_count

    def __str__(self) -> str:
        return 'null_count' if self.null_count else 'count'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute counts for each column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        df = pd.DataFrame()
        if dataset.type_of_target == 'continuous':
            for cnt in CountStatistic.count_to_compute:
                self.null_count = cnt
                data = []
                columns = []
                for col in dataset.X.columns:
                    if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                        columns.append(col)
                        value = dataset.X[col].isna().sum() if self.null_count else dataset.X[col].count()
                        data.append(value)
                df = pd.concat([df, pd.DataFrame([data], index=[str(self)], columns=columns)])
            return df

        class_labels = list(pd.unique(dataset.y))
        for cnt in CountStatistic.count_to_compute:
            self.null_count = cnt
            data = []
            columns = []
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    for label in ['all'] + class_labels:
                        columns.append(f"{col}_{label}")
                        if label == 'all':
                            value = dataset.X[col].isna().sum() if self.null_count else dataset.X[col].count()
                        else:
                            values = dataset.X.loc[dataset.y == label][col]
                            value = values.isna().sum() if self.null_count else values.count()
                        data.append(value)
            df = pd.concat([df, pd.DataFrame([data], index=[str(self)], columns=columns)])
        return df

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
