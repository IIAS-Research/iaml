"""[STATISTIC] Summary Table."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class SummaryTableStatistic(Statistic):
    """[STATISTIC] Summary Table."""

    name: str = "Summary Table"
    _description: str = textwrap.dedent("""\
        Summary table reports global dataset properties.
        """)
    _description_long: str = textwrap.dedent("""\
        Summary table reports global dataset properties: number of rows, number of
        usable columns, total missing values, and memory usage.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'summary_table'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute global summary table for the dataset."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        columns = [
            col for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

        if columns:
            data_frame = dataset.X[columns]
            n_missing_total = int(data_frame.isna().sum().sum())
            memory = int(data_frame.memory_usage(deep=True).sum())
        else:
            n_missing_total = 0
            memory = 0

        data = [[
            dataset.X.shape[0],
            len(columns),
            n_missing_total,
            memory,
        ]]
        return pd.DataFrame(
            data,
            index=[str(self)],
            columns=["n_rows", "n_cols", "n_missing_total", "memory"],
        )

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
