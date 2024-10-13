"""
[STEP] Standard Scaler
"""
import textwrap
from sklearn.preprocessing import StandardScaler
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from ...data_type import DataType

@is_step('normalize')
class ActStandardScaler(Actionable):
    """
    [STEP] Standard Scaler
    """
    name = "Standard Scaler"
    description = textwrap.dedent('''\
        StandardScaler helps computers understand complex data by transforming 
        it into numbers centered around zero with a standard deviation of one.''')
    description_long = textwrap.dedent('''\
        StandardScaler is a machine learning technique used to standardize numerical 
        features by removing the mean and scaling to unit variance. 
        It works by calculating the mean and standard deviation for each feature in the training data, 
        then transforming all values such that they have a mean of zero and a variance of one. 
        This transformation helps to center data and is particularly useful in algorithms that assume 
        normality of features, like many machine learning models.''')
    
    def __init__(self):
        self.configuration:dict = {}
        self.columns:list[str] = None
        self.scaler:StandardScaler = None
    
    
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
            self.scaler = StandardScaler()
            self.scaler.fit(values)
        else: 
            self.scaler = None
        return self
        
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard scaler

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
