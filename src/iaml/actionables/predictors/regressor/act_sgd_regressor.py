"""[STEP] SGD Regressor"""
from typing import Any
import textwrap
from sklearn.linear_model import SGDRegressor
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActSGDRegressor(Predictor):
    """[STEP] SGD Regressor"""

    name: str = "SGD Regressor"
    _usage: str = "Use when you need a fast linear baseline on large tabular data, e.g., vs ActElasticNetRegressor. Applicable to scaled numeric features with a continuous target. Avoid when strong nonlinearities or best accuracy is needed; prefer ActCatBoostRegressor or ActExtraTreesRegressor."
    _description: str = textwrap.dedent('''\
        SGDRegressor is a machine learning algorithm that models
        the relationship between input features and a continuous output variable
        using stochastic gradient descent.''')
    _description_long: str = textwrap.dedent('''\
        SGDRegressor is a type of linear model that models the relationship
        between input features and a continuous output variable using stochastic gradient descent.
        It works by iteratively updating the model parameters in the direction of the negative
        gradient of the loss function with respect to the parameters, using a single example
        at a time.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 1951,
            'name': 'A Stochastic Approximation Method',
            'authors': [
                'Herbert Robbins',
                'Sutton Monro'    
            ],
            'doi': 'https://doi.org/10.1214/aoms/1177729586',
            'publisher': 'The annals of Mathematical Statistics Vol.22 No.3 page 400--407'
        }
    ]
    def __init__(self):
        self.configuration = {
            'alpha': {
                'description': 'Constant that multiplies the regularization term.',
                'default': 0.0001,
                'range': [1e-07, 0.1]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            },
            'epsilon': {
                'description': 'Epsilon in the epsilon-insensitive loss functions',
                'default': 0.1,
                'range': [1e-05, 0.1]
            },
            'eta0': {
                'description': textwrap.dedent('''\
                    The initial learning rate for the ‘constant’, ‘invscaling’
                    or ‘adaptive’ schedules.'''),
                'default': 0.01,
                'range': [1e-07, 0.1]
            },
            'l1_ratio': {
                'description': 'The Elastic Net mixing parameter',
                'default': 0.15,
                'range': [1e-09, 1.0]
            },
            'power_t': {
                'description': 'The exponent for inverse scaling learning rate.',
                'default': 0.25,
                'range': [1e-05, 1.0]
            },
            'average': {
                'description': textwrap.dedent('''\
                    When set to True, computes the averaged SGD weights across
                    all updates and stores the result in the coef_ attribute.'''),
                'default': False
            },
            'loss': {
                'description': 'The loss function to be used.',
                'default': "squared_error",
                'categorical': ["squared_error",
                                "huber",
                                "epsilon_insensitive",
                                "squared_epsilon_insensitive"]
            },
            'penalty': {
                'description': 'The penalty (aka regularization term) to be used.',
                'default': "l2",
                'categorical': ["l1", "l2", "elasticnet"]
            }
        }

        self.model: SGDRegressor = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = SGDRegressor(**self.passthrough_parameters())

        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
