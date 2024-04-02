"""
[STEP] Learn :  Logistic Regression Classifier
"""
from sklearn.linear_model import LogisticRegression
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('predictor', 'tabular', 'fast_predictor', 'classifier')
class ActLogisticRegression(Predictor):
    """
    [STEP] Learn :  Logistic Regression Classifier
    """
    name = "Learn : Logistic Regression Classifier"
    def __init__(self):
        self.configuration:dict = {
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'penalty': {
                'description': 'Specify the norm of the penalty',
                'default': 'l2',
                'categorical': ['l2', None]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            }
        }
        self.model:LogisticRegression = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit Logistic Regression on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = LogisticRegression(
            # n_jobs=-1,
            max_iter=500,
            class_weight='balanced',
            **self.passthrough_parameters()
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
