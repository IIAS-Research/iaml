"""
[STEP] Learn :  Random Forest Regressor
"""
from sklearn.ensemble import RandomForestRegressor
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActRandomForestRegressor(Predictor):
    """
    [STEP] Learn :  Random Forest Regressor
    """
    name = "Learn : Random Forest Regressor" 
    def __init__(self):
        self.configuration:dict = {
            # 'max_depth': { # Disable before probably better with no limit in regression
            #     'description': 'Max depth of each tree',
            #     'default': 15,
            #     'range': [1, 100]
            # },
            'n_estimators': {
                'description': 'Number of threes',
                'default': 100,
                'range': [1, 500]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
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
            },
            'bootstrap': {
                'description': 'Whether bootstrap samples are used when building trees. \
                    If False, the whole dataset is used to build each tree.',
                'default': False
            },
            'criterion': {
                'description': 'The function to measure the quality of a split.',
                'default': "squared_error",
                'categorical': ["poisson", "friedman_mse", "absolute_error", "squared_error"]
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
        self.model = RandomForestRegressor(**self.passthrough_parameters())
        
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
