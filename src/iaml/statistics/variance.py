"""[STATISTIC] Variance."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class VarianceStatistic(Statistic):
    """[STATISTIC] Variance."""

    name: str = "Variance"
    _description: str = textwrap.dedent("""\
        Variance measures unbiased variance for numerical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Variance measures unbiased variance for numerical columns (normalized by N-1),
        per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'variance'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute variance for numerical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
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
                            data.append(dataset.X[col].var())
                        else:
                            data.append(dataset.X.loc[dataset.y == label][col].var())
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
