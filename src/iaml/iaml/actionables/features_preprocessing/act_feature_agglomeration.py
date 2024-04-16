"""
[STEP] Decompose features with FeatureAgglomeration
"""
import pandas as pd
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActFeatureAgglomeration(Actionable):
    """
    [STEP] Agglomerate features with FeatureAgglomeration
    """
    name="Agglomerate features with FeatureAgglomeration"
    
    def __init__(self):
        self.configuration:dict = {
            'n_clusters': {
                'description': 'The number of clusters to find',
                'default': 25,
                'range': [2, 400]
                },
            'metric': {
                'description': 'Metric used to compute the linkage.',
                'default': 'euclidean',
                'categorical': ['euclidean']
                # 'categorical': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
                },
            'linkage': {
                'description': 'Which linkage criterion to use.',
                'default': 'ward',
                'categorical': ['ward', 'complete', 'average', 'single']
                # 'categorical': ['ward', 'complete', 'average', 'single']
                },
            'pooling_func': {
                'description': 'Which linkage criterion to use.',
                'default': np.mean,
                'categorical': [np.mean, np.median, np.max]
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
        self.configure('n_clusters', min(self.get_config('n_clusters'), dataset.X.shape[1]))
        
        self.preprocessor = FeatureAgglomeration(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        
        return self
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Apply FeatureAgglomeration

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
