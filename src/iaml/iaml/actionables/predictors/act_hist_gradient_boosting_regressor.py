"""
[STEP] Learn : HistGradient Boosting Regressor
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

# @is_step('predictor', 'tabular', 'regressor')
class ActHistGradientBoostingRegressor(Predictor):
    """
    [STEP] Learn : HistGradient Boosting Regressor
    """
    name = "Learn : HistGradient Boosting Regressor"
    refs = [
        {
            'year': 2006,
            'name': 'Gaussian Processes for Machine Learning',
            'authors': [
                'Carl Edward Rasmussen',
                'Christopher K. I. Williams'
            ],
            'doi': '',
            'publisher': 'MIT Press 2006'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'l2_regularization': {
                'description': 'The L2 regularization parameter. \
                    Use 0 for no regularization (default).',
                'default': 1e-10,
                'range': [1e-10, 1.0]
                },
            'quantile': {
                'description': 'If loss is “quantile”, this parameter specifies which quantile to \
                    be estimated and must be between 0 and 1.',
                'default': 0.5,
                'range': [0.1, 1.0]
                },
            'learning_rate': {
                'description': 'The learning rate, also known as shrinkage.',
                'default': 0.1,
                'range': [0.01, 1.0]
                },
            'max_leaf_nodes': {
                'description': 'The maximum number of leaves for each tree.',
                'default': 31,
                'range': [3, 2048]
                },
            'min_samples_leaf': {
                'description': 'The minimum number of samples per leaf.',
                'default': 20,
                'range': [1, 200]
                },
            'loss': {
                'description': 'The loss function to use in the boosting process.',
                'default': "squared_error",
                'categorical': ["absolute_error", "poisson", "quantile", "squared_error"]
                },
            'n_iter_no_change': {
                'description': 'Used to determine when to “early stop”.',
                'default': 4,
                'range': [2, 15]
                },
            'tol': {
                'description': 'The absolute tolerance to use when comparing \
                    scores during early stopping',
                'default': 1e-4,
                'range': [1e-8, 1e-2]
                },
            'max_depth': {
                'description': 'The maximum depth of each tree',
                'default': 10,
                'range': [8, 25]
                }
            }
        self.model:HistGradientBoostingRegressor = None
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit HistGradientBoostingRegressor on Candidate.dataset

        Args:
            dataset (Candidate): Fit data

        Returns:
            Fitted step
        """
        self.model = HistGradientBoostingRegressor(early_stopping=True,
                                                   **self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    
    def suitable(self, dataset:Dataset) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return dataset.type_of_target == 'continuous'
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
