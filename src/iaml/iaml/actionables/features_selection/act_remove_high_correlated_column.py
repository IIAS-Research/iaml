"""
[STEP] Remove High Correlated Column
"""
import numpy as np
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_selection')
class ActRemoveHighCorrelatedColumn(Actionable):
    """
    [STEP] Remove High Correlated Column
    """
    name = 'Remove High Correlated Column'
    description = 'Remove columns which correlation with other columns is higher than {threshold}.'
    description_long = ''''''
    def __init__(self):
        self.configuration:dict = {
            'threshold': {
                'description': 'If two columns is correlated over this value, only one \
                    will be kept',
                'default': 0.9
            }
        }
        self.to_drop:list[str] = None
    
    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find high correlated columns to drop

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        # Compute correlation matrix 
        corr_matrix = dataset.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
        # Find features with above-threshold correlation
        # self.to_drop = list(to_drop.keys())
        corr = { c: (upper[upper[c] >= self.get_config('threshold')].index) for c in upper.columns }
        self.to_drop = [ c for c, v in corr.items() if len(v) > 0 ]

        self.explanations = [
            f"""Dropped column **`{c}`** because it was too correlated with
                {", ".join([ f"**`{i}`**" for i in corr[c] ])}."""
            for c in self.to_drop
        ]
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Drop high correlated column

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return X.drop(self.to_drop, axis=1)

        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
