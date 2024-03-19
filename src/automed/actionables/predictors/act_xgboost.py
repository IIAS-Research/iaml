"""
[STEP] Learn :  XGBoost
"""
from sklearn.ensemble import GradientBoostingClassifier
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
class ActXGBoost(Predictor):
    """
    [STEP] Learn :  XGBoost
    """
    name = "Learn : XGBoost"
    def __init__(self):
        self.configuration:dict = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'learning_rate': {
                'description': 'Learning rate',
                'default': 1.0,
                'range': [0.000000001, 5]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, 500]
            }
        }
        self.model:GradientBoostingClassifier = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit XgBoost on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = GradientBoostingClassifier(**self.model_parameters())
            
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
