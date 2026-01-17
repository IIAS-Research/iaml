"""[STEP]  Extra Trees Regressor"""
import textwrap
from typing import Any
from sklearn.ensemble import ExtraTreesRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor')
class ActExtraTreesRegressor(Predictor):
    """[STEP]  Extra Trees Regressor"""

    name: str = "Extra Trees Regressor"
    _description: str = textwrap.dedent('''\
        ExtraTreesRegressor is a machine learning algorithm that makes predictions
        for regression tasks by combining the outputs of multiple decision trees.''')
    _description_long: str = textwrap.dedent('''\
        ExtraTreesRegressor is a type of ensemble learning algorithm
        that belongs to the family of decision tree-based models.
        It works by building multiple decision trees, where each tree is trained on a
        random subset of the input features and a random subset of the training data.
        At prediction time, the algorithm aggregates the outputs of all the
        decision trees to make a final prediction.''')
    _usage: str = "Use when you want a strong nonparametric tabular regressor, often a better default than ActDecisionTreeRegressor or ActAdaBoostRegressor. Applicable to numeric or mixed features with continuous targets. Avoid when data is tiny, very sparse/high-dimensional, or a linear/transparent model is required."
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Extremely randomized trees',
            'authors': [
                'Pierre Geurts',
                'Damien Ernst',
                'Louis Wehenkel'
            ],
            'doi': 'https://doi.org/10.1007/s10994-006-6226-1',
            'publisher': 'Machine Learning Vol. 63 page 3--42'
        }
    ]
    def __init__(self):
        self.configuration = {
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, 100]
            },
            'min_samples_leaf': {
                'description': 'The minimum number of samples required to be at a leaf node.',
                'default': 1,
                'range': [1, 15]
            },
            'max_features': {
                'description': 'The number of features to consider when looking for the best split',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'min_samples_split': {
                'description': 'The minimum number of samples required to split an internal node',
                'default': 2,
                'range': [2, 20]
            },
            'bootstrap': {
                'description': textwrap.dedent('''\
                    Whether bootstrap samples are used when building trees. If
                    False, the whole dataset is used to build each tree.'''),
                'default': False
            },
            'criterion': {
                'description': 'The function to measure the quality of a split.',
                'default': "squared_error",
                'categorical': ["poisson", "friedman_mse", "absolute_error", "squared_error"]
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
        self.model: ExtraTreesRegressor = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = ExtraTreesRegressor(**self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
