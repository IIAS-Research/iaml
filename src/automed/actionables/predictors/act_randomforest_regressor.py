"""
[STEP] Learn :  Random Forest Regressor
"""
from sklearn.ensemble import RandomForestRegressor
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActRandomForestRegressor(Predictor):
    """
    [STEP] Learn :  Random Forest Regressor
    """
    name = "Learn : Random Forest Regressor" 
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model:RandomForestRegressor = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Random Forest regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = RandomForestRegressor(**self.model_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def suitable(self, candidate: Candidate) -> bool:
        return candidate.dataset.type_of_target in ['continuous']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
