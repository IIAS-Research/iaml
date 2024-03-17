
"""
[STEP] Learn :  KNN
"""
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from ...predictor import Predictor
from ...candidate import Candidate
from ...dataset import Dataset
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActKNN(Predictor):
    """
    [STEP] Learn :  KNN
    """
    name = "Learn : KNN"
    def __init__(self):
        self.configuration:dict = {
            'metric': {
                'description': 'Can be minkowski or manhattan',
                'default': 'minkowski',
                'categorical': ['minkowski', 'manhattan']
            },
            'n_neighbors': {
                'description': 'Number of neighbors',
                'default': 5,
                'range': [1, 200]
            }
        }
        self.model:KNeighborsClassifier = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Knn classifier on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = KNeighborsClassifier(
            n_neighbors = min(self.get_config('n_neighbors'), dataset.X.shape[0]),
            metric = self.get_config('metric')
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def suitable(self, candidate) -> bool:
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
