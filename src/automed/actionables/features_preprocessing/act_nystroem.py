"""
[STEP] Decompose features with Nystroem
"""
import pandas as pd
import numpy as np
from sklearn.kernel_approximation import Nystroem
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActNystroem(Actionable):
    """
    [STEP] Approximate with Nystroem
    """
    name="Approximate with Nystroem"
    
    def __init__(self):
        self.configuration:dict = {
            'kernel': {
                'description': 'Kernel map to be approximated.',
                'default': 'rbf',
                'categorical': ["poly", "rbf", "sigmoid", "cosine", "chi2"]
                },
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [50, 10000]
                },
            'coef0': {
                'description': 'Zero coefficient for polynomial and sigmoid kernels.',
                'default': 0.0,
                'range': [-1.0, 1.0]
                },
            'degree': {
                'description': 'Degree of the polynomial kernel.',
                'default': 3,
                'range': [2, 5]
                },
            'gamma': {
                'description': 'Gamma parameter for the RBF, laplacian, polynomial, \
                    exponential chi2 and sigmoid kernels.',
                'default': 0.1,
                'range': [3.06e-05, 8.0]
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
        
        self.preprocessor = Nystroem(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply Nystroem

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