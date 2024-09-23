"""
[STEP] Decompose features with SelectPercentile
"""

import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActSelectPercentile(Actionable):
    """
    [STEP] Preprocess with SelectPercentile
    """
    name="Preprocess with SelectPercentile"
    description = '''SelectPercentile is a tool that helps choose important features from a 
        group of variables by looking at how well each one predicts the outcome.'''
    description_long = '''SelectPercentile is a feature selection technique used in machine learning. 
        It works by assigning scores to each feature based on how well it predicts the outcome. 
        Then, it selects only the top-scoring percentage of features. 
        This helps reduce the number of variables while keeping the most informative ones.'''
    def __init__(self):
        self.configuration:dict = {
            'score_func': {
                'description': 'function taking two arrays X and y, \
                    and returning a pair of arrays',
                'default': chi2,
                'categorical': [chi2, f_classif]
                },
            'percentile': {
                'description': 'Percent of features to keep.',
                'default': 50.0,
                'range': [1.0, 99.0]
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
        
        self.preprocessor = SelectPercentile(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X, dataset.y)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply SelectPercentile

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))
        
    def suitable(self, dataset:Dataset) -> bool:
        # Negative values are not supported
        return not((dataset.X < 0).any().any()) \
            and dataset.type_of_target in \
                ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
