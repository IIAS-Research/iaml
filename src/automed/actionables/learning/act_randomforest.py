"""
[STEP] Learn :  Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('learning', 'tabular')
class ActRandomForest(Predictor):
    """
    [STEP] Learn :  Random Forest
    """
    name = "Learn : Random Forest" 
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, float('inf')]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model:RandomForestClassifier = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Random forest on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = RandomForestClassifier(max_depth=self.get_config('max_depth'),
                                            random_state=self.get_config('random_state'),
                                            n_estimators=self.get_config('n_estimators'))
        
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
    
    
    def suitable(self, candidate) -> bool:
        return candidate.dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
