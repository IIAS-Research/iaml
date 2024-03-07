"""
[STEP] Transform string column to date
"""
import pandas as pd
import numpy as np
from ...actionable import Actionable
from ...data_type import DataType
from ...output import Output, Input
from ...step import is_step, runner

@is_step('cleaning')
class ActSplitDate(Actionable):
    """
    [STEP] Transform string column to date
    """
    name = "Transform string column to date"
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[str] = None
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Find columns to split

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.columns = input_data.dataset.get_columns_names_by_type(DataType.DATE)

        input_data.pipeline.add_explanation(self, [
            f'Split date column **`{c}`** into year, month, weekday, hour, minute and second.'
            for c in self.columns
        ])

        return input_data.add_transform(self)
            
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

    
        
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # medium priority
    