"""[STATISTIC] Outlier Count (IQR)."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class OutlierCountIQRStatistic(Statistic):
    """[STATISTIC] Outlier Count (IQR)."""

    name: str = "Outlier Count (IQR)"
    _description: str = textwrap.dedent("""\
        Outlier count measures how many values fall outside 1.5*IQR.
        """)
    _description_long: str = textwrap.dedent("""\
        Outlier count measures how many values fall outside 1.5*IQR,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'outlier_count_iqr'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute outlier counts (1.5*IQR rule) for each numerical column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        def count_outliers(values: pd.Series) -> int:
            values = values.dropna()
            if values.empty:
                return 0
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr):
                return 0
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return int(((values < lower) | (values > upper)).sum())

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    if dataset.columns_types[col][1] == DataType.NUMERIC:
                        data.append(count_outliers(dataset.X[col]))
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
                            values = dataset.X[col]
                        else:
                            values = dataset.X.loc[dataset.y == label][col]
                        data.append(count_outliers(values))
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
