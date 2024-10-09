"""
[STEP] Learn :  Random Forest Regressor
"""
from sklearn.ensemble import RandomForestRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step
import textwrap

@is_step('predictor', 'tabular', 'regressor')
class ActRandomForestRegressor(Predictor):
    """
    [STEP] Learn :  Random Forest Regressor
    """
    name = "Learn : Random Forest Regressor"
    description = textwrap.dedent('''\
        RandomForestRegressor is a machine learning algorithm that 
        models the relationship between input features and a continuous output 
        variable using a collection of decision trees.''')
    description_long = textwrap.dedent('''\
        RandomForestRegressor is a type of ensemble learning algorithm 
        that models the relationship between input features and a continuous output variable 
        using a collection of decision trees. It works by building multiple decision trees on 
        random subsets of the input features and data, and then averaging the predictions of each 
        tree to make the final prediction.''')
    
    refs = [
        {
            'year': 2001,
            'name': 'Random Forests',
            'authors': ['Leo Breiman'],
            'doi': 'https://doi.org/10.1023/A:1010933404324',
            'publisher': 'Machine Learning Vol.45 page 5--32'
        },
        {
            'year': 2006,
            'name': 'Extremely Randomized Trees',
            'authors': ['Pierre Geurts', 'Damien Ernst', 'Lous Wehenkel'],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol.63 page 5--42'
        }
    ]
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
