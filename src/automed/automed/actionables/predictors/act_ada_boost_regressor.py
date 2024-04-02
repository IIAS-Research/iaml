"""
[STEP] Learn :  AdaBoost Regressor
"""
from sklearn.ensemble import AdaBoostRegressor
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActAdaBoostRegressor(Predictor):
    """
    [STEP] Learn :  AdaBoost Regressor
    """
    name = "Learn : AdaBoost Regressor" 
    def __init__(self):
        self.configuration:dict = {
            'learning_rate': {
                'description': 'Weight applied to each regressor at each boosting iteration',
                'default': 0.1,
                'range': [0.01, 2.0]
            },
            'loss': {
                'description': 'The loss function to use when updating the weights \
                    after each boosting iteration.',
                'default': "linear",
                'categorical': ["linear", "square", "exponential"]
            },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 50,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }
        self.model:AdaBoostRegressor = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit AdaBoost Regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = AdaBoostRegressor(**self.passthrough_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def suitable(self, candidate) -> bool:
        return candidate.dataset.type_of_target == 'continuous'

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
