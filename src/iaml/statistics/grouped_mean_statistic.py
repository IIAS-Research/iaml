"""[STATISTIC] Grouped mean."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class GroupedMeanStatistic(Statistic):
    """[STATISTIC] Grouped mean."""

    name: str = "Grouped Mean"
    _description: str = textwrap.dedent("""\
        Grouped mean measures per-class mean offsets from the overall mean.
        """)
    _description_long: str = textwrap.dedent("""\
        Grouped mean measures numerical column means per class as offsets from the overall mean.
        The overall mean is reported under the `_all` column suffix for each feature.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'grouped_mean'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute grouped mean offsets for numerical columns."""
        if dataset.type_of_target in ['survival', 'continuous']:
            return pd.DataFrame()

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not numeric_columns:
            return pd.DataFrame()

        overall_means = dataset.X[numeric_columns].mean()
        grouped_means = dataset.X[numeric_columns].groupby(dataset.y).mean()
        class_labels = list(pd.unique(dataset.y))

        data = []
        columns = []
        for col in numeric_columns:
            overall = overall_means.get(col)
            for label in ['all'] + class_labels:
                columns.append(f"{col}_{label}")
                if label == 'all':
                    data.append(overall)
                else:
                    class_mean = grouped_means.at[label, col] if label in grouped_means.index else pd.NA
                    if pd.isna(class_mean) or pd.isna(overall):
                        data.append(None)
                    else:
                        data.append(class_mean - overall)

        return pd.DataFrame([data], index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target not in ['survival', 'continuous']
