"""
[STEP] Drop Textual Column
"""
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...data_type import DataType
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('cleaning')
class ActDropTextualColumn(Actionable):
    """
    [STEP] Drop Textual Column
    """
    name = 'Drop textual columns'
    description = 'Drop textual columns.'
    
    def __init__(self):
        self.columns_to_drop:list[str] = None
    
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to drop

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.columns_to_drop = (dataset
            .get_columns_names_by_type([DataType.TEXT, DataType.SHORT_TEXT]))

        self.explanations = [
            f'Dropped column **`{c}`**.' for c in self.columns_to_drop
        ]

        return self    

    def transform(self, x) -> pd.DataFrame:
        """
        Drop columns.

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return x.drop(self.columns_to_drop, axis=1)
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0 # Last cleaning action
