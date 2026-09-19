"""[STATISTIC] Quantile."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class QuantileStatistic(Statistic):
    """[STATISTIC] Quantile."""

    name: str = "Quantile"
    _description: str = textwrap.dedent("""\
        Quantile measures percentile values of numerical columns.
        """)
    _description_long: str = textwrap.dedent("""\
        Quantile measures percentile values of numerical columns, per class for classification.
        """)
    refs: list[dict] = []

    quantile_to_compute = [0.1, 0.25, 0.50, 0.75, 0.90]

    def __init__(self, percentile: float = 0.5) -> None:
        self.percentile = percentile

    def __str__(self) -> str:
        return f'quantile_{self.percentile}'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute quantiles for numerical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        class_labels = list(pd.unique(dataset.y))
        df = pd.DataFrame()
        for quantile in QuantileStatistic.quantile_to_compute:
            self.percentile = quantile
            data = []
            columns = []
            for col in dataset.X.columns:
                if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                    for label in ['all'] + class_labels:
                        columns.append(f"{col}_{label}")
                        if dataset.columns_types[col][1] == DataType.NUMERIC:
                            if label == 'all':
                                data.append(dataset.X[col].quantile(self.percentile))
                            else:
                                data.append(dataset.X.loc[dataset.y == label][col].quantile(self.percentile))
                        else:
                            data.append(None)
            df = pd.concat([df, pd.DataFrame([data], index=[str(self)], columns=columns)])
        return df

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
