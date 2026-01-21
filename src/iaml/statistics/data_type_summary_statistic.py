"""[STATISTIC] Data Type Summary."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class DataTypeSummaryStatistic(Statistic):
    """[STATISTIC] Data Type Summary."""

    name: str = "Data Type Summary"
    _description: str = textwrap.dedent("""\
        Data type summary counts column types and ratios.
        """)
    _description_long: str = textwrap.dedent("""\
        Data type summary reports the number of numeric, categorical, text, and date
        columns, along with ratios relative to the total number of columns.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'data_type_summary'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute data type counts and ratios for the dataset."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        total_columns = len(dataset.X.columns)
        n_numeric = len(dataset.get_columns_names_by_type(DataType.NUMERIC))
        n_categorical = len(dataset.get_columns_names_by_type(DataType.CATEGORICAL))
        n_text = len(dataset.get_columns_names_by_type([DataType.TEXT, DataType.SHORT_TEXT]))
        n_date = len(dataset.get_columns_names_by_type(DataType.DATE))

        if total_columns:
            ratio_numeric = n_numeric / total_columns
            ratio_categorical = n_categorical / total_columns
            ratio_text = n_text / total_columns
            ratio_date = n_date / total_columns
        else:
            ratio_numeric = 0.0
            ratio_categorical = 0.0
            ratio_text = 0.0
            ratio_date = 0.0

        data = [[
            n_numeric,
            n_categorical,
            n_text,
            n_date,
            ratio_numeric,
            ratio_categorical,
            ratio_text,
            ratio_date,
        ]]
        columns = [
            'n_numeric',
            'n_categorical',
            'n_text',
            'n_date',
            'ratio_numeric',
            'ratio_categorical',
            'ratio_text',
            'ratio_date',
        ]
        return pd.DataFrame(data, index=[str(self)], columns=columns)

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
