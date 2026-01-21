"""[STATISTIC] Cardinality Ratio."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class CardinalityRatioStatistic(Statistic):
    """[STATISTIC] Cardinality Ratio."""

    name: str = "Cardinality Ratio"
    _description: str = textwrap.dedent("""\
        Cardinality ratio measures the ratio of unique values to total rows.
        """)
    _description_long: str = textwrap.dedent("""\
        Cardinality ratio measures the ratio of unique values to total rows,
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'cardinality_ratio'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute cardinality ratios for each column."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            n_rows = int(dataset.X.shape[0])
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    n_unique = dataset.X[col].nunique(dropna=True)
                    ratio = n_unique / n_rows if n_rows else 0.0
                    data.append(ratio)
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
                    n_unique = values.nunique(dropna=True)
                    ratio = n_unique / n_rows if n_rows else 0.0
                    data.append(ratio)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
