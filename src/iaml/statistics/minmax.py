"""[STATISTIC] Min / Max."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class BoundStatistic(Statistic):
    """[STATISTIC] Bound."""

    name: str = "Bound"
    _description: str = textwrap.dedent("""\
        Bound measures extreme values (min/max) of numerical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Bound measures extreme values (min/max) of numerical columns, per class for classification.
        """)
    refs: list[dict] = []

    bound_to_compute = ['min', 'max']

    def __init__(self, bound: str = 'min') -> None:
        self.bound = bound

    def __str__(self) -> str:
        return self.bound

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute min/max for numerical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))
        df = pd.DataFrame()
        for bound in BoundStatistic.bound_to_compute:
            self.bound = bound
            data = []
            columns = []
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    for label in ['all'] + class_labels:
                        columns.append(f"{col}_{label}")
                        if dataset.columns_types[col][1] == DataType.NUMERIC:
                            if label == 'all':
                                value = getattr(dataset.X[col], bound)()
                            else:
                                value = getattr(dataset.X.loc[dataset.y == label][col], bound)()
                            data.append(value)
                        else:
                            data.append(None)
            df = pd.concat([df, pd.DataFrame([data], index=[str(self)], columns=columns)])
        return df

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
