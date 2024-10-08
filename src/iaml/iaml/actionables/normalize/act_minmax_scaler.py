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
import textwrap

@is_step('normalize')
class ActMinMaxScaler(Actionable):
    """
    [STEP] Min Max Scaler
    """
    name = textwrap.dedent("Min Max Scaler")
    description = textwrap.dedent('''MinMaxScaler is a tool that helps computers understand complex relationships 
        between things by turning them into simpler numbers within a fixed range.''')
    description_long = textwrap.dedent('''MinMaxScaler is a machine learning technique used to scale numerical 
        features to a fixed range, typically between zero and one. 
        It works by finding the minimum and maximum values for each feature in the training data, 
        then scaling all values to fall between those extremes. 
        This transformation helps ensure that all features are on the same scale, 
        which can improve the performance of many machine learning algorithms.''')
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[str] = None
        self.scaler:MinMaxScaler = None
    
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Find columns to scale and fit scaler

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if self.columns:
            values = dataset.X[self.columns]
            self.scaler = MinMaxScaler()
            self.scaler.fit(values)
        else: 
            self.scaler = None
        return self
        
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply min max scaler

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        if self.scaler:
            X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
