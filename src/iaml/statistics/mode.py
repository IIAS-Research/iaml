"""[STATISTIC] Mode."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class ModeStatistic(Statistic):
    """[STATISTIC] Mode."""

    name: str = "Mode"
    _description: str = textwrap.dedent("""\
        Mode measures the most common values of each column.
        """)
    _description_long: str = textwrap.dedent("""\
        Mode measures the most common values of each column, per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'mode'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute mode for each column."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))
        data = []
        columns = []
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if label == 'all':
                        data.append(dataset.X[col].mode().to_list())
                    else:
                        data.append(dataset.X.loc[dataset.y == label][col].mode().to_list())
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
