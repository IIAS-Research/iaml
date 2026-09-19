"""[STATISTIC] Coefficient of variation."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class CoefVariationStatistic(Statistic):
    """[STATISTIC] Coefficient of variation."""

    name: str = "Coef Variation"
    _description: str = textwrap.dedent("""\
        Coefficient of variation measures relative dispersion (std/mean).
        """)
    _description_long: str = textwrap.dedent("""\
        Coefficient of variation measures relative dispersion (std/mean),
        optionally per class for classification.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'coef_variation'

    def _coef_variation(self, values: pd.Series) -> float | None:
        mean = values.mean()
        if pd.isna(mean) or mean == 0:
            return None
        stdev = values.std()
        if pd.isna(stdev):
            return None
        return stdev / mean

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute coefficient of variation for numerical columns."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        data = []
        columns = []

        if dataset.type_of_target == 'continuous':
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    columns.append(col)
                    if dataset.columns_types[col][1] == DataType.NUMERIC:
                        data.append(self._coef_variation(dataset.X[col]))
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
                        data.append(self._coef_variation(values))
                    else:
                        data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
