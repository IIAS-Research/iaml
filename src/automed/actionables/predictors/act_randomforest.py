"""
[STEP] Learn :  Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular')
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
        self.model:RandomForestClassifier = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Random forest on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = RandomForestClassifier(**self.model_parameters())
        
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
