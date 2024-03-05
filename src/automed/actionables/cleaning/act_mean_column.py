"""
[STEP] Fill missing values with mean
"""
import pandas as pd
from ...actionable import Actionable
from ...output import Output, Input
from ...step import is_step, runner
from ...data_type import DataType


@is_step('cleaning')
class ActMeanColumn(Actionable):
    """
    [STEP] Fill missing values with mean
    """
    name = 'Fill missing values with mean'
    description = 'Fills missing values with the mean of non-missing values when the proportion of empty rows is lower than {empty_threshold}.'

    def __init__(self):
        self.columns:list[str] = None
        self.configurations = [{
            'empty_threshold': {
                'description': 'Column with less or equal proportion of empty row will\
                    be fill with mean value. 1 will always fill void values',
                'default': 0.5
            }
        }]
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument  
        self.columns = []
        explain = []

        for column in input_data.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input_data.dataset.X[column]
            nan_values_count = values.isnull().sum()

            if nan_values_count > 0 and nan_values_count / len(values) <= self.get_config('empty_threshold'):
                self.columns.append((column, values.mean()))
                explain.append((nan_values_count, len(values), nan_values_count / len(values) * 100))

        input_data.pipeline.add_explanation(self, [
            f'Filled missing values of column **`{c}`** with **{mean:.2f}** because **{v[0]}** out of **{v[1]}** values (**{v[2]:.2f}**%) were missing.'
            for (c, mean), v in zip(self.columns, explain)
        ])
        
        return input_data.add_transform(self)
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Fill NA values with mean

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        for name, mean in self.columns:
            X[name].fillna(mean, inplace=True)

        return X
        
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 1-(input_data.dataset.X.isnull().sum().min()/len(input_data.dataset.X))
