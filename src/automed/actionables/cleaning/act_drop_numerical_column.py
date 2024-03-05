"""
[STEP] Drop Numerical Column
"""
import pandas as pd
from ...actionable import Actionable
from ...data_type import DataType
from ...output import Output, Input
from ...step import is_step, runner

@is_step('cleaning')
class ActDropNumericalColumn(Actionable):
    """
    [STEP] Drop Numerical Column
    """
    name = 'Drop numerical columns'
    description = """Drop numerical columns where the proportion of empty rows
        in the dataset is higher than {empty_threshold}."""
    
    def __init__(self):
        self.columns_to_drop:list[str] = None
        self.configuration:dict = {
            'empty_threshold': {
                'description': 'Column with more or equal proportion of empty row \
                    will dropped. 1 will drop all columns',
                'default': 0.5
            }
        }
    
    @runner
    def run(self, input_data: Input, callback: callable = None) -> Output: # pylint: disable=unused-argument
        """
        Find columns to drop

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.columns_to_drop = []
        explain = []

        for column in input_data.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input_data.dataset.X[column]
            nan_values_count = values.isnull().sum()

            if nan_values_count / len(values) >= self.get_config('empty_threshold'):
                self.columns_to_drop.append(column)
                explain.append((
                    nan_values_count,
                    len(values),
                    nan_values_count / len(values) * 100,
                ))

        input_data.pipeline.add_explanation(self, [
            f"""Dropped column **`{c}`** because **{v[0]}** values out of
                **{v[1]}** (**{v[2]:.2f}%**) are empty."""
            for c, v in zip(self.columns_to_drop, explain)
        ])

        return input_data.add_transform(self)
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Drop numerical column

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return X.drop(self.columns_to_drop, axis=1)
        
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0 # Last cleaning action
