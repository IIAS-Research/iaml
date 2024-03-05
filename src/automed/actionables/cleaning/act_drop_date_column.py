"""
[STEP] Find and drop date column
"""
import pandas as pd
from ...actionable import Actionable
from ...data_type import DataType
from ...output import Output, Input
from ...step import is_step, runner

@is_step('cleaning')
class ActDropDateColumn(Actionable):
    """
    Find and drop data column
    """
    name = "Drop date columns"
    description = 'Drop date columns.'
    
    def __init__(self):
        self.columns_to_drop:list[str] = None
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Find column to drop

        Args:
            input_data (Input): Data to fit on
            callback (callable, optional): Call after each run. Defaults to None.

        Returns:
            Output: Transformed output (with updated pipeline)
        """
        self.columns_to_drop = input_data.dataset.get_columns_names_by_type(DataType.DATE)

        input_data.pipeline.add_explanation(self, [
            f'Dropped column **`{c}`**.' for c in self.columns_to_drop
        ])

        return input_data.add_transform(self)
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Drop all date column of input dataset

        Args:
            x (pd.DataFrame): Dataset to transform

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
