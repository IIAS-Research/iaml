"""[STATISTIC] Duplicate Rows."""
from __future__ import annotations

import textwrap
import pandas as pd

from ..dataset import Dataset
from ..data_type import DataType
from ..statistic import Statistic


class DuplicateRowStatistic(Statistic):
    """[STATISTIC] Duplicate Rows."""

    name: str = "Duplicate Rows"
    _description: str = textwrap.dedent("""\
        Duplicate rows reports the number and ratio of duplicated rows.
        """)
    _description_long: str = textwrap.dedent("""\
        Duplicate rows counts duplicated rows based on usable columns and
        provides the ratio relative to the total number of rows.
        """)
    refs: list[dict] = []

    def __str__(self) -> str:
        return 'duplicate_rows'

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute number and ratio of duplicated rows."""
        if dataset.type_of_target == 'survival':
            return pd.DataFrame()

        columns = [
            col for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

        n_rows = int(dataset.X.shape[0])
        if columns:
            data_frame = dataset.X[columns]
            n_duplicate = int(data_frame.duplicated().sum())
        else:
            n_duplicate = 0

        ratio = n_duplicate / n_rows if n_rows else 0.0

        data = [[n_duplicate, ratio]]
        return pd.DataFrame(
            data,
            index=[str(self)],
            columns=["n_duplicate_rows", "ratio_duplicate_rows"],
        )

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Does this statistic apply to the dataset?"""
        return dataset.type_of_target != 'survival'
