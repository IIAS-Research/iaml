"""
[STEP] Preprocess with PowerTransformer
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActPowerTransformer(Actionable):
    """
    [STEP] Preprocess with PowerTransformer
    """
    name="Preprocess with PowerTransformer"
    
    def __init__(self):
        self.configuration:dict = {
            'method': {
                'description': 'The power transform method.',
                'default': 'yeo-johnson',
                'categorical': ['yeo-johnson', 'box-cox']
                },
            'standardize': {
                'description': 'Set to True to apply zero-mean, unit-variance normalization to the transformed output.',
                'default': True
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
        
        self.preprocessor = PowerTransformer(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply PowerTransformer

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
