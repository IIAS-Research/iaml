"""[STATISTIC] Violin."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd
from scipy import stats

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class ViolinStatistic(Statistic):
    """[STATISTIC] Violin."""

    name: str = "Violin"
    _description: str = textwrap.dedent("""\
        Violin statistics summarize target distributions per categorical value.
        """)
    _description_long: str = textwrap.dedent("""\
        Violin statistics summarize target distributions per categorical value
        to support violin-style plots.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'violin'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute violin statistics for categorical features."""
        if dataset.type_of_target != 'continuous':
            return pd.DataFrame()

        data = []
        columns = []
        for col in dataset.X.columns:
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT):
                columns.append(col)
                if dataset.columns_types[col][1] == DataType.CATEGORICAL:
                    categorical_values = dataset.X[col].unique()
                    stats_dict = {}
                    for cat in categorical_values:
                        cond = (dataset.X[col] == cat).to_numpy()
                        tmp = dataset.y[cond]
                        if len(tmp) > 1:
                            kernel = stats.gaussian_kde(tmp)
                            support = np.linspace(min(tmp), max(tmp), 100)
                            density = kernel(support)
                            quartiles = np.percentile(tmp, [25, 50, 75])
                            stats_dict[cat] = {
                                'density': density,
                                'support': support,
                                'quartiles': quartiles
                            }
                    data.append(stats_dict)
                else:
                    data.append(None)
        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target == 'continuous'
