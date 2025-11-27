"""[STEP] Gradient Boosting Regressor"""
import textwrap
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor

from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'regressor', 'minimal_predictor')
class ActGBoostRegressor(Predictor):
    """[STEP] Gradient Boosting Regressor"""

    name: str = "Gradient Boosting Regressor"
    _description: str = textwrap.dedent('''\
        GradientBoostingRegressor learns an ensemble of weak learners (decision trees)
        in a stage-wise manner to minimize the prediction error on continuous targets.''')
    _description_long: str = textwrap.dedent('''\
        Gradient Boosting is an additive modeling technique where each subsequent tree
        attempts to correct the residuals of the previous ensemble. The regressor is
        robust to different loss formulations and usually provides a strong baseline
        for tabular regression tasks.''')
    refs: list[dict[str, Any]] = [
        {
            'name': 'Greedy Function Approximation: A Gradient Boosting Machine',
            'year': 2001,
            'authors': ['Jerome H. Friedman'],
            'doi': 'https://doi.org/10.1214/aos/1013203451',
            'publisher': 'The Annals of Statistics, Vol.29, No.5'
        }
    ]

    def __init__(self):
        self.configuration = {
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
                'range': [1e-8, 5.0]
            },
            'n_estimators': {
                'description': 'Number of estimators',
                'default': 100,
                'range': [1, 500]
            },
            'loss': {
                'description': 'Loss function to optimize.',
                'default': "squared_error",
                'categorical': ['squared_error', 'absolute_error', 'huber', 'quantile']
            },
            'criterion': {
                'description': 'Split quality metric',
                'default': "friedman_mse",
                'categorical': ['friedman_mse', 'squared_error']
            },
            'min_samples_leaf': {
                'description': 'Minimum samples at leaf nodes.',
                'default': 1,
                'range': [1, 15]
            },
            'max_features': {
                'description': 'Fraction of features per split.',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'min_samples_split': {
                'description': 'Minimum samples to split nodes.',
                'default': 2,
                'range': [2, 20]
            }
        }
        self.model: GradientBoostingRegressor = None

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.model = GradientBoostingRegressor(**self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous'

    def priorize(self, candidate: Candidate = None) -> float:  # pylint: disable=unused-argument
        return 0.5
