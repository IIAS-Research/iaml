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
    
    def __init__(self):
        self.configuration:dict = {
            'score_func': {
                'description': 'unction taking two arrays X and y, \
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
        
    def suitable(self, candidate: Candidate) -> bool:
        # Negative values are not supported
        return not((candidate.dataset.X < 0).any().any() or (candidate.dataset.y < 0).any()) \
            and candidate.dataset.type_of_target in \
                ['binary', 'multiclass',  'multilabel-indicator']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
