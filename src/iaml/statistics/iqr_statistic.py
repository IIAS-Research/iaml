"""[STATISTIC] IQR."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class IQRStatistic(Statistic):
    """[STATISTIC] IQR."""

    name: str = "IQR"
    _description: str = textwrap.dedent("""\
        IQR measures dispersion as Q3 minus Q1 for numerical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        IQR measures dispersion as Q3 minus Q1 for numerical columns, per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'iqr'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute IQR for each numerical column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    if dataset.columns_types[col][1] == DataType.NUMERIC:
                        q3 = dataset.X[col].quantile(0.75)
                        q1 = dataset.X[col].quantile(0.25)
                        data.append(q3 - q1)
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
                            series = dataset.X[col]
                        else:
                            series = dataset.X.loc[dataset.y == label][col]
                        q3 = series.quantile(0.75)
                        q1 = series.quantile(0.25)
                        data.append(q3 - q1)
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
