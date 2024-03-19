"""
[STEP] Decompose features with PCA
"""
from copy import deepcopy
import pandas as pd
from sklearn.decomposition import PCA
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step



@is_step('features_preprocessing')
class ActPCA(Actionable):
    """
    [STEP] Decompose features with PCA
    """
    name="Decompose features with PCA"
    
    def __init__(self):
        self.configuration:dict = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 0.999,
                'range': [0.5, 0.999]
                },
            'random_state': {
                'description': 'Random State',
                'default': 42
                }
            }
        
        self.optimizable = True
        self.preprocessor = None


    def fit(self, dataset:Dataset) -> Actionable:
        """
        Find columns to convert

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.preprocessor = PCA(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))
        
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5