"""[STATISTIC] Skewness."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class SkewnessStatistic(Statistic):
    """[STATISTIC] Skewness."""

    name: str = "Skewness"
    _description: str = textwrap.dedent("""\
        Skewness measures the asymmetry of numerical distributions.
        """)
    _description_long: str = textwrap.dedent("""\
        Skewness measures the asymmetry of numerical distributions, per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'skewness'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute skewness for numerical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        if dataset.type_of_target == 'continuous':
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))
        data = []
        columns = []
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if dataset.columns_types[col][1] == DataType.NUMERIC:
                        if label == 'all':
                            data.append(dataset.X[col].skew())
                        else:
                            data.append(dataset.X.loc[dataset.y == label][col].skew())
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
