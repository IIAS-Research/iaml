"""
[STEP] Find and drop date column
"""
import pandas as pd
from ...actionable import Actionable
from ...data_type import DataType
from ...candidate import Candidate
from ...dataset import Dataset
from ...decorators.all import is_step

@is_step('cleaning')
class ActDropDateColumn(Actionable):
    """
    Find and drop data column
    """
    name = 'Remove date columns'
    description = 'Remove all columns containing Date from the dataset'
    description_long = '''Remove all columns containing Data from the dataset
        This step is used to clean the dataset in order to perform other actions later on 
        that can't be applied to date columns.'''
    
    def __init__(self):
        self.columns_to_drop:list[str] = None
    
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find column to drop

        Args:
            dataset (Dataset): Data to fit on

        Returns:
            Candidate: Transformed candidate (with updated pipeline)
        """
        self.columns_to_drop = dataset.get_columns_names_by_type(DataType.DATE)

        self.explanations = [
            f'Dropped column **`{c}`**.' for c in self.columns_to_drop
        ]
        
        return self
        
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Drop all date column of candidate dataset

        Args:
            x (pd.DataFrame): Dataset to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return X.drop(self.columns_to_drop, axis=1) 
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0 # Last cleaning action
