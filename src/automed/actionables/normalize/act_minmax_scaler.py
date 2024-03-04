"""
[STEP] Min Max Scaler
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from ...actionable import Actionable
from ...output import Output, Input
from ...step import is_step, runner
from ...data_type import DataType


@is_step('normalize')
class ActMinMaxScaler(Actionable):
    """
    [STEP] Min Max Scaler
    """
    name = "Min Max Scaler"
    def __init__(self):
        self.configurations:list[dict] = [{}]
        self.columns:list[str] = None
        self.scaler:MinMaxScaler = None
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Find columns to scale and fit scaler

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        self.columns = input_data.dataset.get_columns_names_by_type(DataType.NUMERIC)
        values = input_data.dataset.X[self.columns]
        self.scaler = MinMaxScaler()
        self.scaler.fit(values)

        return input_data.add_transform(self)
        
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply min max scaler

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
