"""
[STEP] Learn : KNN
"""
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from ...predictor import Predictor
from ...candidate import Candidate
from ...dataset import Dataset
from ...decorators.all import is_step



@is_step('learning', 'tabular', 'fast_learning')
class ActKNNRegressor(Predictor):
    """
    [STEP] Learn : KNN
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
        self.model:KNeighborsRegressor = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Knn regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = KNeighborsRegressor(
            n_neighbors = self.get_config('n_neighbors'),
            metric = self.get_config('metric')
            )
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def predict(self, X:pd.DataFrame) -> list[float]:

        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        return self.model.predict(X)
    
    
    def suitable(self, candidate: Candidate) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return candidate.dataset.type_of_target in ['continuous']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
