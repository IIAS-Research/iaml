"""[STATISTIC] Median Absolute Deviation."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class MADStatistic(Statistic):
    """[STATISTIC] Median Absolute Deviation."""

    name: str = "Median Absolute Deviation"
    _description: str = textwrap.dedent("""\
        Median absolute deviation measures the typical absolute deviation from the median.
        """)
    _description_long: str = textwrap.dedent("""\
        Median absolute deviation measures the typical absolute deviation from the median,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'mad'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute median absolute deviation for each numerical column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        def mad(series: pd.Series) -> float:
            median = series.median()
            return (series - median).abs().median()

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    if dataset.columns_types[col][1] == DataType.NUMERIC:
                        data.append(mad(dataset.X[col]))
                    else:
                        data.append(None)
            return pd.DataFrame([data], index=[str(self)], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if dataset.columns_types[col][1] == DataType.NUMERIC:
                        if label == 'all':
                            data.append(mad(dataset.X[col]))
                        else:
                            data.append(mad(dataset.X.loc[dataset.y == label][col]))
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
