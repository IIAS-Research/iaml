"""
[STEP] Decompose features with RBFSampler
"""
import pandas as pd
import numpy as np
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step
from sklearn.kernel_approximation import RBFSampler


@is_step('features_preprocessing')
class ActRBFSampler(Actionable):
    """
    [STEP] Approximate with RBFSampler
    """
    name="Approximate with RBFSampler"
    
    def __init__(self):
        self.configuration:dict = {
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [50, 10000]
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
        Fit Features agglomerations

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        
        self.preprocessor = RBFSampler(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply RBFSampler

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