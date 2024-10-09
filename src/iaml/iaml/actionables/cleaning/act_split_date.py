"""
[STEP] Transform string column to date
"""
import textwrap
import pandas as pd
import numpy as np
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step
import textwrap


@is_step('cleaning')
class ActSplitDate(Actionable):
    """
    [STEP] Transform string column to date
    """
    name = textwrap.dedent('Create Date Elements columns')
    descrption = textwrap.dedent('''Transform a textual date column into multiple columns
        for day, month, year, hour, minute, second''')
    description_long = textwrap.dedent('''Transform a textual date column into multiple columns for 
        day, month, year, hour, minute, second.
        Exemple:
+---------------------+----------+------------+-----------+-----------+----------+----------+
| date                | date_day | date_month | date_year | date_hour | date_min | date_sec |
+---------------------+----------+------------+-----------+-----------+----------+----------+
| 2024-01-15 12:31:27 | 15       | 01         | 2024      | 12        | 31       | 27       |
+---------------------+----------+------------+-----------+-----------+----------+----------+''')
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[str] = None
    
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to split

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.columns = dataset.get_columns_names_by_type(DataType.DATE)

        self.explanations = [
            f'Split date column **`{c}`** into year, month, weekday, hour, minute and second.'
            for c in self.columns
        ]

        return self
            
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Split dates columns into columns -> weekday, mount, year, hour, minute, second

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
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
    
    def suitable(self, dataset:Dataset) -> bool:
        return bool(dataset.get_columns_names_by_type(DataType.DATE))
        
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # medium priority
    