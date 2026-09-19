"""[STEP] Transform string column to date"""
import textwrap
import pandas as pd
import numpy as np
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('cleaning')
class ActSplitDate(Actionable):
    """[STEP] Transform string column to date"""

    name: str = 'Create Date Elements columns'
    _usage: str = "Use when you want to expand a date column into components rather than ActDropDateColumn. Applicable to date-typed columns with usable timestamps. Avoid when dates are already split or you plan to drop them via ActDropDateColumn."
    _descrption: str = textwrap.dedent('''\
        Transform a textual date column into multiple columns
        for day, month, year, hour, minute, second''')
    _description_long: str = textwrap.dedent('''\
        Transform a textual date column into multiple columns for 
        day, month, year, hour, minute, second.
        Exemple:
        +-------------------+----------+------------+-----------+-----------+----------+----------+
        |date               | date_day | date_month | date_year | date_hour | date_min | date_sec |
        +-------------------+----------+------------+-----------+-----------+----------+----------+
        |2024-01-15 12:31:27| 15       | 01         | 2024      | 12        | 31       | 27       |
        +-------------------+----------+------------+-----------+-----------+----------+----------+
        ''')

    def __init__(self):
        self.columns: list[str] = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.columns = dataset.get_columns_names_by_type(DataType.DATE)

        self.explanations = [
            f'Split date column **`{c}`** into year, month, weekday, hour, minute and second.'
            for c in self.columns
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Split dates columns into columns (weekday, mount, year, hour, minute, second).

        :param pd.DataFrame x: DataFrame to transform.
        :return: Transformed dataset.
        """
        for column in self.columns:
            # Day
            X[column + '_weekday'] = X[column].dt.dayofweek.replace(np.NaN, -1)
            X[column + '_month'] = X[column].dt.month.replace(np.NaN, -1)
            X[column + '_year'] = X[column].dt.year.replace(np.NaN, -1)

            # Hour
            X[column + '_hour'] = X[column].dt.hour.replace(np.NaN, -1)
            X[column + '_minute'] = X[column].dt.minute.replace(np.NaN, -1)
            X[column + '_second'] = X[column].dt.second.replace(np.NaN, -1)

        return X

    def suitable(self, dataset: Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.DATE))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
