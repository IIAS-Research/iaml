"""
[STEP] Learn :  XGBoost Regressor
"""
from sklearn.ensemble import GradientBoostingRegressor
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActXGBoostRegressor(Predictor):
    """
    [STEP] Learn :  XGBoost Regressor
    """
    name = "Learn : XGBoost Regressor"
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
                'default': 0.1,
                'range': [0.000000001, 5]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, 500]
            },
            'loss': {
                'description': 'The loss function to use in the boosting process.',
                'default': "squared_error",
                'categorical': ['squared_error', 'absolute_error', 'huber', 'quantile']
            },
            'criterion': {
                'description': 'The function to measure the quality of a split',
                'default': "friedman_mse",
                'categorical': ['friedman_mse', 'squared_error']
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 15]
            },
            'max_features': {
                'description': 'The number of features to consider when looking for the best split',
                'default': 1,
                'range': [0.1, 1]
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node',
                'default': 2,
                'range': [2, 20]
            }
        }
        self.model:GradientBoostingRegressor = None
        
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit XgBoost regressor on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = GradientBoostingRegressor(**self.passthrough_parameters())
        
        self.model.fit(dataset.X, dataset.y)
        
        return self
    
    def suitable(self, dataset:Dataset) -> bool:
        return dataset.type_of_target in ['continuous']

    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
