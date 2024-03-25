"""
[STEP] Decompose features with KernelPCA
"""
import pandas as pd
from sklearn.decomposition import KernelPCA
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActKernelPCA(Actionable):
    """
    [STEP] Decompose features with KernelPCA
    """
    name="Decompose features with KernelPCA"
    
    def __init__(self):
        self.configuration:dict = {
            'kernel': {
                'description': 'Kernel used for PCA.',
                'default': 'rbf',
                'categorical': ['poly', 'rbf', 'sigmoid', 'cosine']
                },
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [10, 2000]
                },
            'coef0': {
                'description': 'Independent term in poly and sigmoid kernels. \
                    Ignored by other kernels.',
                'default': 1.0,
                'range': [-1.0, 1.0]
                },
            'degree': {
                'description': 'Degree for poly kernels. Ignored by other kernels.',
                'default': 3,
                'range': [2, 5]
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
        
        self.preprocessor = KernelPCA(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply KernelPCA

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
