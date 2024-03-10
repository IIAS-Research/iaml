"""
[STEP] Min Max Scaler
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType


@is_step('normalize')
class ActMinMaxScaler(Actionable):
    """
    [STEP] Min Max Scaler
    """
    name = "Min Max Scaler"
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[str] = None
        self.scaler:MinMaxScaler = None
    
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Find columns to scale and fit scaler

        Args:
            candidate (Candidate): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Candidate: Transformed candidate
        """
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        values = dataset.X[self.columns]
        self.scaler = MinMaxScaler()
        self.scaler.fit(values)

        return self
        
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

    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
