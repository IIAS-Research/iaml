"""[STATISTIC] Entropy."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class EntropyStatistic(Statistic):
    """[STATISTIC] Entropy."""

    name: str = "Entropy"
    _description: str = textwrap.dedent("""\
        Entropy measures the distribution uncertainty for categorical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Entropy measures the distribution uncertainty for categorical columns,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'entropy'

    def _entropy(self, series: pd.Series) -> float:
        values = series.dropna()
        if values.empty:
            return 0.0
        counts = values.value_counts()
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        return float(-(probs * np.log2(probs)).sum())

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute entropy for categorical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                    columns.append(col)
                    data.append(self._entropy(dataset.X[col]))
            return pd.DataFrame([data], index=[str(self)], columns=columns)

        class_labels = list(pd.unique(dataset.y))
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                for label in ['all'] + class_labels:
                    columns.append(f"{col}_{label}")
                    if label == 'all':
                        values = dataset.X[col]
                    else:
                        values = dataset.X.loc[dataset.y == label][col]
                    data.append(self._entropy(values))
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
